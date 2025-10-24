use std::sync::Arc;

use libheif_rs::{
    Channel, ColorSpace, CompressionFormat, EncoderQuality, HeifContext, LibHeif, RgbChroma,
};
use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, DeviceCreateInfo, DeviceFeatures, QueueCreateInfo, QueueFlags},
    format::{ClearColorValue, Format},
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::{self, GpuFuture},
};

mod shader;

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
            enabled_features: DeviceFeatures {
                shader_float16: true,
                shader_int16: true,
                storage_buffer16_bit_access: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("Failed to create device");

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let width = 4000;
    let height = 3000;

    let rgb_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_SFLOAT,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let r_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..width * height * 2).map(|_| 0u8),
    )
    .unwrap();

    let g_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..width * height * 2).map(|_| 0u8),
    )
    .unwrap();

    let b_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..width * height * 2).map(|_| 0u8),
    )
    .unwrap();

    let compute_shader = shader::finishing::load(device.clone())
        .expect("Failed to create shader module")
        .entry_point("main")
        .unwrap();
    let stage = PipelineShaderStageCreateInfo::new(compute_shader);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("Failed to create compute pipeline");

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, ImageView::new_default(rgb_image.clone()).unwrap()),
            WriteDescriptorSet::buffer(1, r_buffer.clone()),
            WriteDescriptorSet::buffer(2, g_buffer.clone()),
            WriteDescriptorSet::buffer(3, b_buffer.clone()),
        ],
        [],
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

    command_buffer_builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([1.0, 1.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(rgb_image.clone())
        })
        .unwrap()
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap();

    unsafe {
        command_buffer_builder
            .dispatch([width / 8, height / 8, 1])
            .unwrap();
    }

    let command_buffer = command_buffer_builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let mut image =
        libheif_rs::Image::new(width, height, ColorSpace::Rgb(RgbChroma::C444)).unwrap();

    image.create_plane(Channel::R, width, height, 12).unwrap();
    image.create_plane(Channel::G, width, height, 12).unwrap();
    image.create_plane(Channel::B, width, height, 12).unwrap();

    let planes = image.planes_mut();

    planes
        .r
        .unwrap()
        .data
        .copy_from_slice(&*r_buffer.read().unwrap());

    planes
        .g
        .unwrap()
        .data
        .copy_from_slice(&*g_buffer.read().unwrap());

    planes
        .b
        .unwrap()
        .data
        .copy_from_slice(&*b_buffer.read().unwrap());

    let lib_heif = LibHeif::new();
    let mut context = HeifContext::new().unwrap();
    let mut encoder = lib_heif
        .encoder_for_format(CompressionFormat::Hevc)
        .unwrap();

    encoder.set_quality(EncoderQuality::LossLess).unwrap();
    context.encode_image(&image, &mut encoder, None).unwrap();

    context.write_to_file("image.heic").unwrap();
}
