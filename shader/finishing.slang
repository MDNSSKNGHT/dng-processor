// RGBA half precision normalized RGB image
RWTexture2D<half4> RGB;

// Output structured buffers for R, G, B channels
RWStructuredBuffer<uint16_t> R_buffer;
RWStructuredBuffer<uint16_t> G_buffer;
RWStructuredBuffer<uint16_t> B_buffer;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadID: SV_DispatchThreadID) {
  uint width, height;
  RGB.GetDimensions(width, height);

  uint2 coords = threadID.xy;

  // if (coords.x >= width || coords.y >= height)
  //   return;

  half4 pixel = RGB[coords];

  uint index = coords.y * width + coords.x;

  R_buffer[index] = uint16_t(pixel.r * 4095.0h);
  G_buffer[index] = uint16_t(pixel.g * 4095.0h);
  B_buffer[index] = uint16_t(pixel.b * 4095.0h);
}
