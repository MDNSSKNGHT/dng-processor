use std::sync::Arc;

use libheif_rs::{
    Channel, ColorSpace, CompressionFormat, EncoderQuality, HeifContext, LibHeif, RgbChroma,
};
use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    format::{ClearColorValue, Format},
    image::{Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{self, GpuFuture},
};

fn main() {
    let library = VulkanLibrary::new().expect("Failed to find local Vulkan library");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Failed to enumarate physical devices")
        .next()
        .expect("Failed to find physical device");

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::COMPUTE)
        })
        .expect("Failed to find a compute queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Failed to create device");

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let width = 4000;
    let height = 3000;

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_UINT,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..width * height * 4 * 2).map(|_| 0u8),
    )
    .unwrap();

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Just clear color to purple
    command_buffer_builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Uint([1000, 50, 1000, 4095]),
            ..ClearColorImageInfo::image(image.clone())
        })
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buffer.clone(),
        ))
        .unwrap();

    let command_buffer = command_buffer_builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let mut image =
        libheif_rs::Image::new(width, height, ColorSpace::Rgb(RgbChroma::HdrRgbaLe)).unwrap();

    image
        .create_plane(Channel::Interleaved, width, height, 12)
        .unwrap();

    let planes = image.planes_mut();

    let plane_interleaved = planes.interleaved.unwrap();

    plane_interleaved
        .data
        .copy_from_slice(&*buffer.read().unwrap());

    let lib_heif = LibHeif::new();
    let mut context = HeifContext::new().unwrap();
    let mut encoder = lib_heif
        .encoder_for_format(CompressionFormat::Hevc)
        .unwrap();

    encoder.set_quality(EncoderQuality::LossLess).unwrap();
    context.encode_image(&image, &mut encoder, None).unwrap();

    context.write_to_file("image.heic").unwrap();
}
