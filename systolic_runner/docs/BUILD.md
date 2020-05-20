## Building

To build, first ensure that you have the Risc-V toolchain setup, built with`esp-tools`
(from Chipyard, use `/scripts/build-toolchains.sh esp-tools`).
 
Once you have riscv g++ in your `PATH`, clone this repo and `git submodule update --init --recursive`.
Then run `./build.sh --parallel`. Note that while Microsoft claims cmake might not get the dependency order right for `--parallel`,
in my experience it has worked fine.
This will build with debug symbols – for release mode (`-O3` you can use `./build.sh --config=Release --parallel`).

To build the `image-net` runner, use the same command you used for building ORT, (i.e. if you built in release mode, use `./build.sh --config=Release`)

### Building for Firesim

When building for running with Firesim (as opposed to `spike pk`), you must add the following to the beginning of `runner.cpp` in the imagenet runner to prevent page-fault related issues.

```
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
```

You must also add to the end of `systolic_include.h` in `onnxruntime/core/mlas/lib/systolic/systolic_include.h` the following to ensure that Gemmini flushes the TLB for subsequent runs in a new process.

```
__attribute__((constructor))
void cleargemmini() {
  gemmini_flush(0);
}
```

Please follow the [FireSim documentation](https://docs.fires.im/en/latest/) for instructions on how to use FireSim to run the built binary. When building the FireMarshal workload using buildroot linux, you will need to enable RoCC Extensions by patching `arch/riscv/kernel/process.c` in `firemarshal/riscv-linux` (if this is not done, you may encounter "Unhandled signal 4" trap):

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

## Running

To run using `spike`, please first pull `master` for `riscv-isa-sim` in `esp-tools`.

You will next need to patch `riscv-pk` to no-op the futex and tid syscalls as follows:


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

You can rebuild spike + pk via `./build-spike-pk.sh`

Note that by default, Chipyard adds `chipyard/riscv-tools-install/bin/spike` to your PATH, which does not contain the gemmini extension. Please ensure (manually adding the `esp-tools` build folder to $PATH if needed) that `which spike` instead references the spike from `esp-tools`.

See the README on imagenet runner for documentation on how to use onnxruntime to run an imagenet model. Sample quantized models are provided in the [releases](https://github.com/pranav-prakash/onnxruntime-riscv/releases) tab, or you can perform post-training quantization on a model yourself using the quantization tool in the systolic_runner folder. 

You can also run other models by creating the appropriate runner script that calls into the onnxruntime APIs.
