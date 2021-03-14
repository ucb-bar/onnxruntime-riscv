## Building

### Setting up your Toolchain

To build, first ensure that you have the Risc-V toolchain setup. We recommend using [Chipyard](https://chipyard.readthedocs.io/en/latest/Chipyard-Basics/Initial-Repo-Setup.html) to install the riscv toolchain, but if you prefer you can also install riscv-g++ manually.

Note that there are two "flavors" of the risc-v toolchain: the standard one (riscv-tools), and a variant (esp-tools) with specific support for ucb-bar Gemmini and Hwacha accelerators. In particular, esp-tools has a version of g++ that supports assembling instructions for Hwacha, and has a version of spike that supports the Gemmini and hwacha extensions.

For this project, it actually suffices to use standard (riscv-tools) gcc to build the binaries since emitting Gemmini instructions doesn't need a custom assembler (note that building the optional hwacha backend will need esp-tools, though). But since we would need the spike from `esp-tools` to simulate the binary anyway, we recommend just grabbing `esp-tools` to make things simpler.

To summarize, here are all the ways you can acquire a suitable toolchain:

* Using chipyard (probably easiest, but might be overkill since it installs lots of extra dependencies)
     - Follow the above linked Chipyard docs, making sure to use `./scripts/build-toolchains.sh esp-tools` to install the esp-tools toolchain
     - As mentioned, building `riscv-tools` would also be sufficient to compile onnxruntime, but that version of spike won't be able to simulate Gemmini
* Installing a toolchain manually
     - You can manually install the gnu-toolchain, riscv-pk, and riscv-isa-sim found within this [esp-tools](https://github.com/ucb-bar/chipyard/tree/master/toolchains/esp-tools) folder
     - Note that the [esp-tools repo](https://github.com/ucb-bar/esp-tools) is OUT OF DATE and should not be used. (See https://github.com/pranav-prakash/onnxruntime-riscv/issues/27)


**tl;dr** Using the esp-tools risc-v toolchain from Chipyard is the best option for most people

### Building this repo

Once you have riscv g++ in your `PATH`, clone this repo and `git submodule update --init --recursive`.

Ensure that the `systolic_params.h` matches the gemmini config you wish to run against (that is, its contents should match the auto-generated `gemmini_params.h` file from the Gemmini build). While you should likely not need to touch `systolic_include.h`, if the Gemmini ISA has changed recently and this repo has not yet been updated to match, `systolic_include.h` will need to be updated as well -- symptoms of a Gemmini-version mismatch include freezing or incorrect outputs when running the unit tests or entire networks.

The Gemmini data-type to build against (int8 or fp32) can be selected in `CMakeLists.txt`

```
option(onnxruntime_SYSTOLIC_FP32 "If Systolic is enabled, whether to use for fp32 ops" ON)
option(onnxruntime_SYSTOLIC_INT8 "If Systolic is enabled, whether to use for int8 ops" OFF) 
```

For training, `fp32` is needed.

Then run `./build.sh --parallel`. Note that while Microsoft claims cmake might not get the dependency order right for `--parallel`,
in my experience it has worked fine.
This will build with debug symbols – for release mode (`-O3`) you can use `./build.sh --config=Release --parallel`.

To build the `image-net` runner (located in the `systolic` folder at the root), use the exact same command you used for building ORT, (i.e. if you built in release mode and without training, you would use `./build.sh --config=Release`)

To build with training support, add `--enable_training` when building ORT and the imagenet runner.

**TL;DR**: `git submodule update --init --recursive` then `./build.sh --config=Release --parallel --enable_training` for both ORT and the model runners

### Building for Firesim

When building for running with Firesim (as opposed to `spike pk` or `qemu`), please build with `--for_firesim` on both the main ORT library and any model runner. This flag will 1) ensure that `mlockall` is performed at process start which prevents page-fault issues from Gemmini accessing a swapped out page 2) flush gemmini on process start. 

Please follow the [FireSim documentation](https://docs.fires.im/en/latest/) for instructions on how to use FireSim to run the built binary. When building the FireMarshal workload using buildroot linux, you may need to enable RoCC Extensions by patching `arch/riscv/kernel/process.c` in `firemarshal/riscv-linux` (if this is not done, you may encounter "Unhandled signal 4" trap):

* Change line 70 (`regs->status |= SR_FS_INITIAL;`) to `regs->status |= SR_FS_INITIAL | SR_XS_INITIAL`).

(You may also encounter "You seem to have the current working directory in your PATH environment variable" error when building buildroot for FireMarshal. In this case, please patch `firemarshal/wlutil/br/buildroot/support/dependencies/dependencies.sh` as follows):

```
 # An empty PATH is technically possible, but in practice we would not
 # even arrive here if that was the case.
 case ":${PATH:-unset}:" in

-(*::*|*:.:*)
-       echo
-       echo "You seem to have the current working directory in your"
-       echo "PATH environment variable. This doesn't work."
-       exit 1
-       ;;
 (*"
 "*)    printf "\n"
        printf "Your PATH contains a newline (\\\n) character.\n"
```

## Running via Spike

You will first need to patch `riscv-pk` to no-op the futex and tid syscalls as follows:


```
--- a/pk/syscall.h
+++ b/pk/syscall.h
@@ -52,6 +52,9 @@
 #define SYS_clock_gettime 113
 #define SYS_set_tid_address 96
 #define SYS_set_robust_list 99
+#define SYS_futex 98
+#define SYS_gettid 178

 #define OLD_SYSCALL_THRESHOLD 1024

 #define SYS_open 1024
 ```
 
```
diff --git a/pk/syscall.c b/pk/syscall.c
@@ -434,6 +434,7 @@ long do_syscall(long a0, long a1, long a2, long a3, long a4, long a5, unsigned l
     [SYS_brk] = sys_brk,
     [SYS_uname] = sys_uname,
     [SYS_getpid] = sys_getpid,
+    [SYS_gettid] = sys_getpid,
     [SYS_getuid] = sys_getuid,
     [SYS_geteuid] = sys_getuid,
     [SYS_getgid] = sys_getuid,

@@ -462,6 +463,7 @@ long do_syscall(long a0, long a1, long a2, long a3, long a4, long a5, unsigned l
     [SYS_chdir] = sys_chdir,
     [SYS_set_tid_address] = sys_stub_nosys,
     [SYS_set_robust_list] = sys_stub_nosys,
+    [SYS_futex] = sys_stub_success,
   };
 ```
 
Finally please also double check that the proxy kernel is patched to enable RoCC extensions (should be done by default), as is shown in commit 
https://github.com/riscv/riscv-pk/commit/c53de08b9ba719f3e7b02fc1a029d194a190da48

You can rebuild pk via:

```
$ mkdir build
$ cd build
$ ../configure --prefix=$RISCV --host=riscv64-unknown-elf
$ make
$ make install
```

Then please pull `master` for the `riscv-isa-sim` in [esp-tools](https://github.com/ucb-bar/chipyard/tree/master/toolchains/esp-tools) so Spike uses the latest Gemmini ISA.

Note that you will want to ensure the `gemmini_params.h` used by spike matches the version used by onnxruntime. I.e., if you built for FP32 Gemmini (check the CMake file as mentioned in the previous section) ensure it matches `systolic_params_fp32.h`. It's also a good idea to change the definition of the `ROUND_NEAR_EVEN` macro to `nearbyint(x)` so the CPU version isn't excessively slow.

Note that by default, Chipyard adds `chipyard/riscv-tools-install/bin/spike` to your PATH, which does not contain the gemmini extension. Please ensure (manually adding the `esp-tools` build folder to $PATH if needed) that `which spike` instead references the spike from `esp-tools`.

See the README on imagenet runner for documentation on how to use onnxruntime to run an imagenet model. Sample quantized models are provided in the [releases](https://github.com/pranav-prakash/onnxruntime-riscv/releases) tab, or you can perform post-training quantization on a model yourself using the quantization tool in the systolic_runner folder. 

You can also run other models by creating the appropriate runner script that calls into the onnxruntime APIs.

## Running via Qemu

Refer to the last paragraph of above.
