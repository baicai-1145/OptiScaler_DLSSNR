// Writes out the frames this pass is arguing about, so questions get settled by measurement.
//
// Every comparison so far has been two separate video captures: different camera paths, different
// exposure, and H.264 in between throwing away exactly the high-frequency temporal detail the argument
// is about. Normalising for all that leaves conclusions that need hedging.
//
// This captures the same frames twice instead. The pass already holds the frame as the upscaler produced
// it and the frame after the model's edit, so both are written for the same run of consecutive frames --
// a perfect control, with nothing varying but the thing under test. Flicker in one against flicker in
// the other is then a direct answer rather than an inference.
//
// Raw, because a codec is the confound. A manifest alongside says how to read them.

#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace capture
{
// Bounded on purpose: this writes hundreds of megabytes and must not be able to fill a drive. Each run
// overwrites the last.
constexpr unsigned int kMaxFrames = 8;

struct Shot
{
    ID3D12Resource* readback = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
    unsigned long long bytes = 0;
};

// Captures a run of consecutive frames of several images at once.
//
// Capture v2: alongside the before/after pair, the model's actual inputs are recorded -- the
// colour the model was shown (post colour-transform), the depth guide, and the motion vector
// guide. Those three are what a reimplementation needs to reproduce the edit; before/after alone
// leaves the guides to be guessed, and they steer the model spatially. Scalar parameters that
// reach the model are written to the manifest alongside.
class FrameCapture
{
  public:
    // Asks for a capture. Ignored if one is already running.
    void request(unsigned int frames)
    {
        if (active_)
            return;

        wanted_ = frames > kMaxFrames ? kMaxFrames : frames;
        captured_ = 0;
        active_ = wanted_ > 0;
    }

    bool isActive() const { return active_; }
    unsigned int progress() const { return captured_; }

    // Records copies of both images for this frame. Both must be in the state given.
    void record(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12Resource* before,
                D3D12_RESOURCE_STATES beforeState, ID3D12Resource* after, D3D12_RESOURCE_STATES afterState)
    {
        if (!active_ || before == nullptr || after == nullptr)
            return;

        // Recording stops the moment the run is complete, and does not resume until write() has
        // released everything.
        //
        // ready_ is set here but acted on eight frames later, because the caller has to let the GPU
        // finish with these copies before mapping them. Without this guard those eight frames each
        // recorded another one: captured_ walked past the end of a vector sized to exactly wanted_,
        // and the garbage read back as a Shot had its pointer handed to CopyTextureRegion. The device
        // was removed and the game went down with nothing in the log -- every single time anyone
        // pressed the button.
        if (ready_ || captured_ >= wanted_)
            return;

        if (!ensure(device, before, after))
        {
            active_ = false;
            return;
        }

        // ensure() sizes the vectors to wanted_. Belt and braces: an index into them is never taken
        // on trust again.
        if (captured_ >= beforeShots_.size() || captured_ >= afterShots_.size())
        {
            ready_ = true;
            return;
        }

        copy(cmd, before, beforeState, beforeShots_[captured_]);
        copy(cmd, after, afterState, afterShots_[captured_]);

        // Guides ride along on the same frame index. They are optional: an older call site that
        // knows nothing about them still produces a valid v1 capture.
        if (captured_ < guideShots_[0].size())
        {
            for (size_t g = 0; g < guideShots_.size(); ++g)
                if (guideShots_[g][captured_].readback != nullptr)
                    copy(cmd, guideResources_[g], guideStates_[g], guideShots_[g][captured_]);
        }

        ++captured_;

        if (captured_ >= wanted_)
            ready_ = true;
    }

    // Declares the guide resources to record alongside before/after, and allocates their readback
    // buffers. Must be called before the first record() of a run, since allocation happens once.
    // All guides are copied from NON_PIXEL_SHADER_RESOURCE, the state every model input is left in.
    void setGuides(ID3D12Device* device, const std::vector<ID3D12Resource*>& resources)
    {
        // Idempotent within a run: the same resources arrive every frame, and re-allocating would
        // leak the buffers already holding earlier frames of the run.
        if (!guideShots_.empty())
            return;

        guideResources_ = resources;
        guideStates_.assign(resources.size(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        guideShots_.resize(resources.size());

        for (size_t g = 0; g < resources.size(); ++g)
        {
            if (resources[g] == nullptr)
                continue;

            guideShots_[g].resize(wanted_);

            D3D12_RESOURCE_DESC desc = resources[g]->GetDesc();
            desc.Format = TypedForCopy(desc.Format);
            guideDescs_.push_back(desc);

            for (unsigned int i = 0; i < wanted_; ++i)
                alloc(device, desc, guideShots_[g][i]);
        }
    }

    // True once every frame has been copied and the GPU has been waited for. The caller does the waiting,
    // since it owns the fence.
    bool readyToWrite() const { return ready_; }

    // Writes what was captured and releases everything. Returns the directory, or an empty string.
    std::string write(const std::filesystem::path& directory)
    {
        if (!ready_)
            return {};

        // A dark frame -- a menu, a loading screen -- measures nothing. Discard the run and quietly
        // re-arm for the same length, so the set that finally lands is of actual gameplay.
        if (captured_ > 0 && isDark(beforeShots_[0]))
        {
            const unsigned int frames = wanted_;
            release();
            ready_ = false;
            active_ = false;
            request(frames);
            return {};
        }

        std::error_code ec;
        std::filesystem::create_directories(directory, ec);

        for (unsigned int i = 0; i < captured_; ++i)
        {
            dump(directory, "before", i, beforeShots_[i]);
            dump(directory, "after", i, afterShots_[i]);

            // v2 guide images. Named for the model input each one carries.
            static const char* guideNames[] = { "model_input", "depth", "motion" };
            for (size_t g = 0; g < guideShots_.size() && g < 3; ++g)
                if (i < guideShots_[g].size() && guideShots_[g][i].readback != nullptr)
                    dump(directory, guideNames[g], i, guideShots_[g][i]);
        }

        writeManifest(directory);
        release();
        return directory.string();
    }

    void release()
    {
        for (auto& s : beforeShots_)
            if (s.readback != nullptr)
                s.readback->Release();

        for (auto& s : afterShots_)
            if (s.readback != nullptr)
                s.readback->Release();

        for (auto& shots : guideShots_)
            for (auto& s : shots)
                if (s.readback != nullptr)
                    s.readback->Release();

        beforeShots_.clear();
        afterShots_.clear();
        guideShots_.clear();
        guideResources_.clear();
        guideStates_.clear();
        guideDescs_.clear();
        scalars_.clear();
        active_ = false;
        ready_ = false;
        captured_ = 0;
    }

    // Scalar parameters to write into the manifest: exactly what the model was told this run.
    void setScalar(const std::string& name, double value) { scalars_.push_back({ name, value }); }

  private:
    bool ensure(ID3D12Device* device, ID3D12Resource* before, ID3D12Resource* after)
    {
        if (!beforeShots_.empty())
            return true;

        beforeShots_.resize(wanted_);
        afterShots_.resize(wanted_);
        beforeDesc_ = before->GetDesc();
        afterDesc_ = after->GetDesc();
        beforeDesc_.Format = TypedForCopy(beforeDesc_.Format);
        afterDesc_.Format = TypedForCopy(afterDesc_.Format);

        for (unsigned int i = 0; i < wanted_; ++i)
        {
            if (!alloc(device, beforeDesc_, beforeShots_[i]) || !alloc(device, afterDesc_, afterShots_[i]))
                return false;
        }

        return true;
    }

    // A copy needs a fully typed format, and the buffer the upscaler writes is occasionally declared
    // typeless. A placed footprint carrying a typeless format makes CopyTextureRegion invalid, and an
    // invalid copy removes the device -- which is what capturing did: it crashed the game every time
    // the output happened to be one of these.
    static DXGI_FORMAT TypedForCopy(DXGI_FORMAT f)
    {
        switch (f)
        {
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        default:
            return f;
        }
    }

    static bool alloc(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, Shot& shot)
    {
        D3D12_RESOURCE_DESC typed = desc;
        typed.Format = TypedForCopy(desc.Format);

        unsigned long long total = 0;
        device->GetCopyableFootprints(&typed, 0, 1, 0, &shot.layout, nullptr, nullptr, &total);
        shot.bytes = total;

        if (total == 0)
            return false;

        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_READBACK;

        D3D12_RESOURCE_DESC buf = {};
        buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buf.Width = total;
        buf.Height = 1;
        buf.DepthOrArraySize = 1;
        buf.MipLevels = 1;
        buf.Format = DXGI_FORMAT_UNKNOWN;
        buf.SampleDesc.Count = 1;
        buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&shot.readback)));
    }

    static void copy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* src, D3D12_RESOURCE_STATES state,
                     Shot& shot)
    {
        if (shot.readback == nullptr)
            return;

        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = src;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = state;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;

        const bool needsTransition = state != D3D12_RESOURCE_STATE_COPY_SOURCE;

        if (needsTransition)
            cmd->ResourceBarrier(1, &b);

        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = shot.readback;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = shot.layout;

        D3D12_TEXTURE_COPY_LOCATION source = {};
        source.pResource = src;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = 0;

        cmd->CopyTextureRegion(&dst, 0, 0, 0, &source, nullptr);

        if (needsTransition)
        {
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b.Transition.StateAfter = state;
            cmd->ResourceBarrier(1, &b);
        }
    }

    // Mean byte value across the image, sampled sparsely. Correct for 8-bit and 10-bit surfaces;
    // float surfaces of dark content read higher, which only means a float capture is never discarded.
    static bool isDark(Shot& shot)
    {
        if (shot.readback == nullptr || shot.bytes == 0)
            return false;

        void* mapped = nullptr;
        D3D12_RANGE range = { 0, (SIZE_T) shot.bytes };

        if (FAILED(shot.readback->Map(0, &range, &mapped)) || mapped == nullptr)
            return false;

        const unsigned char* p = (const unsigned char*) mapped;
        unsigned long long total = 0;
        unsigned long long count = 0;

        for (unsigned long long i = 0; i < shot.bytes; i += 1021)
        {
            total += p[i];
            ++count;
        }

        D3D12_RANGE written = { 0, 0 };
        shot.readback->Unmap(0, &written);
        return count > 0 && total / count < 4;
    }

    static void dump(const std::filesystem::path& dir, const char* which, unsigned int index, Shot& shot)
    {
        if (shot.readback == nullptr)
            return;

        void* mapped = nullptr;
        D3D12_RANGE range = { 0, (SIZE_T) shot.bytes };

        if (FAILED(shot.readback->Map(0, &range, &mapped)) || mapped == nullptr)
            return;

        char name[64];
        std::snprintf(name, sizeof(name), "%s_%02u.raw", which, index);
        const auto path = dir / name;

        if (std::FILE* f = _wfopen(path.wstring().c_str(), L"wb"))
        {
            std::fwrite(mapped, 1, (size_t) shot.bytes, f);
            std::fclose(f);
        }

        D3D12_RANGE written = { 0, 0 };
        shot.readback->Unmap(0, &written);
    }

    void writeManifest(const std::filesystem::path& dir)
    {
        const auto path = dir / "manifest.txt";

        if (std::FILE* f = _wfopen(path.wstring().c_str(), L"wt"))
        {
            std::fprintf(f, "frames %u\n", captured_);
            std::fprintf(f, "before width %llu height %u format %d rowPitch %u\n",
                         (unsigned long long) beforeDesc_.Width, beforeDesc_.Height,
                         (int) beforeDesc_.Format,
                         beforeShots_.empty() ? 0 : beforeShots_[0].layout.Footprint.RowPitch);
            std::fprintf(f, "after width %llu height %u format %d rowPitch %u\n",
                         (unsigned long long) afterDesc_.Width, afterDesc_.Height, (int) afterDesc_.Format,
                         afterShots_.empty() ? 0 : afterShots_[0].layout.Footprint.RowPitch);
            std::fprintf(f, "\nbefore_NN.raw is the frame as the upscaler produced it.\n");
            std::fprintf(f, "after_NN.raw is the same frame once the model's edit was applied.\n");
            std::fprintf(f, "Consecutive frames, same run, so the pair is a control.\n");

            if (!guideDescs_.empty())
            {
                static const char* guideNames[] = { "model_input", "depth", "motion" };
                for (size_t g = 0; g < guideDescs_.size() && g < 3; ++g)
                    std::fprintf(f, "%s width %llu height %u format %d rowPitch %u\n",
                                 guideNames[g], (unsigned long long) guideDescs_[g].Width,
                                 guideDescs_[g].Height, (int) guideDescs_[g].Format,
                                 guideShots_[g].empty() ? 0 : guideShots_[g][0].layout.Footprint.RowPitch);
                std::fprintf(f, "model_input_NN.raw is the colour the model was shown (post transform).\n");
                std::fprintf(f, "depth_NN.raw and motion_NN.raw are the guides the model received.\n");
            }

            for (const auto& s : scalars_)
                std::fprintf(f, "param %s %g\n", s.first.c_str(), s.second);

            std::fclose(f);
        }
    }

    std::vector<Shot> beforeShots_;
    std::vector<Shot> afterShots_;
    std::vector<std::vector<Shot>> guideShots_;
    std::vector<ID3D12Resource*> guideResources_;
    std::vector<D3D12_RESOURCE_STATES> guideStates_;
    std::vector<D3D12_RESOURCE_DESC> guideDescs_;
    std::vector<std::pair<std::string, double>> scalars_;
    D3D12_RESOURCE_DESC beforeDesc_ = {};
    D3D12_RESOURCE_DESC afterDesc_ = {};
    unsigned int wanted_ = 0;
    unsigned int captured_ = 0;
    bool active_ = false;
    bool ready_ = false;
};
} // namespace capture
