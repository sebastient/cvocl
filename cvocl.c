#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <clutils.h>
#include <clprofiler.h>

typedef struct {
    const char* label;
#if __APPLE__
    struct timeval start;
    struct timeval stop;
#else
    struct timespec start;
    struct timespec stop;
#endif
    long runtime;
} profiler_t;

static const char* opencl_strerror(cl_int error);

static void start_profile(profiler_t *p)
{
#if __APPLE__
    gettimeofday(&p->start, NULL);
#else
    clock_gettime(CLOCK_MONOTONIC, &p->start);
#endif
}

static void stop_profile(profiler_t *p)
{
    int64_t start, stop;
    
#if __APPLE__
    gettimeofday(&p->stop, NULL);
#else
    clock_gettime(CLOCK_MONOTONIC, &p->stop);
#endif
    
    start = p->start.tv_sec;
    start *= 1000000;
    
    stop = p->stop.tv_sec;
    stop *= 1000000;
    
#if __APPLE__
    start += p->start.tv_usec;
    stop += p->stop.tv_usec;
#else
    start += p->start.tv_nsec / 1000;
    stop += p->stop.tv_nsec / 1000;
#endif
    
    p->runtime = stop - start;
}

static cl_uint device_selector(CLUDeviceInfo *info, cl_uint num_devices, void* arg)
{
    printf("OpenCL Devices: ");
    for (int i = 0; i < num_devices; i++) {
        printf("%s ", info[i].device_name);
    }
    printf("\n");
    
    return 0;
}

int main(int argc, const char** argv)
{
    cl_int width = 720, height = 480;
	int fd;
	struct stat finfo;
	void *source_image_data;
    void *target_image_data;
    
    CLUZone *cl_zone;
    GError *gerr;
    cl_int err;
    cl_kernel kernel;
    cl_image_format cl_image_format;
    cl_mem cl_source_image, cl_target_image;
    size_t local_work_size[2] = { 16, 16 };
    size_t global_work_size[2] = { 32, 32 };
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { width, height, 1 };
    
    profiler_t profiles[] = {
        { "write image" }, { "kernel" }, { "read image" }, { NULL }
    };
    
    const char *kernels[] = { "copy.cl" };
    
    if (argc != 2) {
		fprintf(stderr, "usage: %s <image>\n", argv[0]);
		return 1;
	}
    
	fd = open(argv[1], O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "failed to open %s: %s\n", argv[1], strerror(errno));
		return 1;
	}
    
	if (fstat(fd, &finfo) == -1) {
		fprintf(stderr, "failed to stat %s: %s\n", argv[1], strerror(errno));
		return 1;
	}
    
	source_image_data = mmap(NULL, finfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (source_image_data == MAP_FAILED) {
		fprintf(stderr, "failed to mmap %s: %s\n", argv[1], strerror(errno));
		return 1;
	}
    
    target_image_data = malloc(finfo.st_size);
    
    cl_zone = clu_zone_new(CL_DEVICE_TYPE_GPU, 1, 0, device_selector, NULL, &gerr);
    if (gerr) g_error("clu_zone_new failed: %s", gerr->message);
    
    clu_program_create(cl_zone, kernels, 1, NULL, &gerr);
    if (gerr) g_error("clu_program_create failed: %s", gerr->message);
    
    kernel = clCreateKernel(cl_zone->program, "copy", &err);
    if (err != CL_SUCCESS) g_error("clCreateKernel failed: %s", opencl_strerror(err));
    
    cl_image_format.image_channel_order = CL_RGBA;
    cl_image_format.image_channel_data_type = CL_UNORM_INT8;
    
    cl_source_image = clCreateImage2D(cl_zone->context,
                                      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                      &cl_image_format,
                                      width, height, 0,
                                      source_image_data,
                                      &err);
    
    if (err != CL_SUCCESS) g_error("failed to create source image: %s", opencl_strerror(err));
    
    cl_target_image = clCreateImage2D(cl_zone->context,
                                      CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                      &cl_image_format,
                                      width, height, 0,
                                      target_image_data,
                                      &err);
    
    if (err != CL_SUCCESS) g_error("failed to create target image: %s", opencl_strerror(err));
    
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_source_image);
    if (err != CL_SUCCESS) g_error("failed to set kernel argument 0: %s", opencl_strerror(err));
    
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_target_image);
    if (err != CL_SUCCESS) g_error("failed to set kernel argument 2: %s", opencl_strerror(err));

    err = clSetKernelArg(kernel, 2, sizeof(cl_int), &width);
    if (err != CL_SUCCESS) g_error("failed to set kernel argument 3: %s", opencl_strerror(err));
    
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &height);
    if (err != CL_SUCCESS) g_error("failed to set kernel argument 4: %s", opencl_strerror(err));
    
    start_profile(&profiles[0]);
    err = clEnqueueWriteImage(cl_zone->queues[0],
                              cl_source_image, CL_TRUE,
                              origin,
                              region,
                              0,
                              0,
                              source_image_data,
                              0, NULL, NULL);
    stop_profile(&profiles[0]);
    if (err != CL_SUCCESS) g_error("failed to write kernel image: %s", opencl_strerror(err));
    
    start_profile(&profiles[1]);
    err = clEnqueueNDRangeKernel(cl_zone->queues[0],
                                 kernel,
                                 2,
                                 NULL,
                                 global_work_size,
                                 local_work_size,
                                 0, NULL, NULL);
    stop_profile(&profiles[1]);
    if (err != CL_SUCCESS) g_error("kernel failure [0x%.08x]: %s", err, opencl_strerror(err));

    start_profile(&profiles[2]);
    err = clEnqueueReadImage(cl_zone->queues[0],
                             cl_target_image,
                             CL_TRUE,
                             origin,
                             region,
                             0,
                             0,
                             target_image_data,
                             0, NULL, NULL);
    stop_profile(&profiles[2]);
    if (err != CL_SUCCESS) g_error("failed to read kernel image: %s", opencl_strerror(err));
    
    for (int i = 0; profiles[i].label; i++) {
        printf("Task %s - %ld usec.\n", profiles[i].label, profiles[i].runtime);
    }
    
    printf("Completed successfully.\n");
    
    clu_zone_free(cl_zone);
    return 0;
}

static const char* opencl_strerror(cl_int error)
{
    switch (error) {
    case CL_SUCCESS:                            return "Success!";
    case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                   return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
    case CL_MAP_FAILURE:                        return "Map failure";
    case CL_INVALID_VALUE:                      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
    case CL_INVALID_PLATFORM:                   return "Invalid platform";
    case CL_INVALID_DEVICE:                     return "Invalid device";
    case CL_INVALID_CONTEXT:                    return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
    case CL_INVALID_SAMPLER:                    return "Invalid sampler";
    case CL_INVALID_BINARY:                     return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
    case CL_INVALID_PROGRAM:                    return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "Invalid global work size";
    case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
    case CL_INVALID_EVENT:                      return "Invalid event";
    case CL_INVALID_OPERATION:                  return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
    default:                                    return "Unknown";
    }
}
