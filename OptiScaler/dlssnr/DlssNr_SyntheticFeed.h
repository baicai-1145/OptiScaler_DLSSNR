// DlssNr_SyntheticFeed.h — Route A oracle probes: replace the game frame with a
// synthetic one before the encode, so the official model can be queried with
// arbitrary inputs (flat fields, impulses, combs, reset sequences).
//
// Wire format (file "dlssnr-probe.bin" beside the DLL):
//   u32 magic 'PRB1', u32 width, u32 height, u32 frames,
//   then frames * width * height * 8 bytes of R16G16B16A16_FLOAT (row-major).
// One file per capture run; the frame fed is frameIndex % frames.
// Presence of the file activates the path (evaluate still runs normally).
// The capture manifest gains scalar "probe=1".
#pragma once

#include <d3d12.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <windows.h>

namespace DlssNr::SyntheticFeed
{

struct State
{
    std::vector<uint8_t> pixels;   // all frames, fp16 RGBA
    unsigned int width = 0, height = 0, frames = 0;
    ID3D12Resource* upload = nullptr;
    ID3D12Resource* texture = nullptr;   // created in the target's format
    unsigned long long served = 0;
    bool failed = false;
};

inline State g_feed;

inline bool EnsureLoaded(unsigned int wantW, unsigned int wantH)
{
    namespace fs = std::filesystem;
    fs::path p = fs::current_path() / "dlssnr-probe.bin";
    std::error_code ec;
    auto stamp = fs::last_write_time(p, ec);
    if (ec || !fs::exists(p, ec))
        return false;
    static fs::file_time_type loadedStamp {};
    static bool loaded = false;
    if (loaded && stamp == loadedStamp && g_feed.width == wantW && g_feed.height == wantH)
        return g_feed.width != 0;
    if (g_feed.upload) { g_feed.upload->Release(); g_feed.upload = nullptr; }
    if (g_feed.texture) { g_feed.texture->Release(); g_feed.texture = nullptr; }
    std::ifstream f(p, std::ios::binary);
    uint32_t hdr[4] {};
    f.read(reinterpret_cast<char*>(hdr), 16);
    if (!f || hdr[0] != 0x31425250 /* 'PRB1' */)
    {
        g_feed.failed = true;
        return false;
    }
    // The feed is authored at the working resolution; if it disagrees we still
    // feed it (scaled nowhere) only when exact — otherwise bail to the game frame.
    g_feed.width = hdr[1]; g_feed.height = hdr[2]; g_feed.frames = hdr[3];
    size_t bytes = (size_t) g_feed.frames * g_feed.width * g_feed.height * 8;
    g_feed.pixels.resize(bytes);
    f.read(reinterpret_cast<char*>(g_feed.pixels.data()), bytes);
    loaded = true; loadedStamp = stamp;
    g_feed.served = 0;
    return g_feed.width == wantW && g_feed.height == wantH;
}

// Creates the GPU-side copy resources. Call with the device + target format.
inline bool EnsureResources(ID3D12Device* device, DXGI_FORMAT fmt)
{
    if (g_feed.texture != nullptr || g_feed.failed)
        return g_feed.texture != nullptr;
    if (g_feed.width == 0)
        return false;
    D3D12_HEAP_PROPERTIES up { D3D12_HEAP_TYPE_UPLOAD };
    size_t rowPitch = ((size_t) g_feed.width * 8 + 255u) & ~255u;
    size_t oneFrame = rowPitch * g_feed.height;
    size_t total = oneFrame * g_feed.frames;
    D3D12_RESOURCE_DESC bd { D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1,
                             DXGI_FORMAT_UNKNOWN, { 1, 0 }, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                             D3D12_RESOURCE_FLAG_NONE };
    if (FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_feed.upload))))
    { g_feed.failed = true; return false; }
    // Map once and stage every frame with row padding now.
    {
        void* map = nullptr;
        D3D12_RANGE r { 0, 0 };
        if (FAILED(g_feed.upload->Map(0, &r, &map)))
        { g_feed.failed = true; return false; }
        auto* dst = static_cast<uint8_t*>(map);
        for (unsigned int fi = 0; fi < g_feed.frames; ++fi)
            for (unsigned int y = 0; y < g_feed.height; ++y)
                memcpy(dst + (size_t) fi * oneFrame + (size_t) y * rowPitch,
                       g_feed.pixels.data() + (((size_t) fi * g_feed.height + y) * g_feed.width * 8),
                       (size_t) g_feed.width * 8);
        g_feed.upload->Unmap(0, nullptr);
    }
    D3D12_HEAP_PROPERTIES def { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC td { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, g_feed.width, g_feed.height,
                             1, 1, fmt, { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_feed.texture))))
    { g_feed.failed = true; return false; }
    // One-time copy of all frames? A texture2D has one frame; simplest correct
    // approach: keep an array texture is overkill — instead copy the CURRENT frame
    // from the buffer each Feed() call into texture (already in COPY_DEST first time).
    return true;
}

// Issues the copy of frame (index % frames) into g_feed.texture and leaves it in
// UAV — the exact state the encode's first Barrier (UAV -> SRV) expects of a
// "fresh upscaler output", so the downstream state machine runs unchanged.
inline ID3D12Resource* Feed(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, DXGI_FORMAT fmt)
{
    if (!EnsureResources(device, fmt))
        return nullptr;
    unsigned int fi = g_feed.frames ? (unsigned int) (g_feed.served % g_feed.frames) : 0;
    size_t rowPitch = ((size_t) g_feed.width * 8 + 255u) & ~255u;
    D3D12_TEXTURE_COPY_LOCATION src {};
    src.pResource = g_feed.upload;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp {};
    fp.Offset = (size_t) fi * rowPitch * g_feed.height;
    fp.Footprint.Format = fmt;
    fp.Footprint.Width = g_feed.width;
    fp.Footprint.Height = g_feed.height;
    fp.Footprint.Depth = 1;
    fp.Footprint.RowPitch = (UINT) rowPitch;
    src.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION dst { g_feed.texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };

    // Texture state on entry: COPY_DEST on the very first call (creation state),
    // UAV on every later call (that is where the previous frame's resolve left it).
    if (g_feed.served != 0)
    {
        D3D12_RESOURCE_BARRIER toCopy { { g_feed.texture, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                          D3D12_RESOURCE_STATE_COPY_DEST } };
        cmd->ResourceBarrier(1, &toCopy);
    }
    cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER toUav { { g_feed.texture, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS } };
    cmd->ResourceBarrier(1, &toUav);
    ++g_feed.served;
    return g_feed.texture;
}

} // namespace
