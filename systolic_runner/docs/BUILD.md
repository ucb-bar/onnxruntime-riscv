To build, first ensure that you have risc-v toolchain setup, built with`esp-tools`
(from Chipyard, use `/scripts/build-toolchains.sh esp-tools`). Note that to run with `spike,` you will need to
patch `pk` to no-op the futex and tid syscalls as follows:

To run this with spike you'll need to patch `pk` to no-op the futex and tid syscall:



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
 
Once you have riscv g++ in your `PATH`, clone this repo and `git submodule update --init --recursive`.
Then run `./build.sh --parallel`. Note that while Microsoft claims cmake might not get the dependency order right for `--parallel`,
in my experience it has worked fine.
This will build with debug symbols – for release mode (`-O3` you can use `./build.sh --config=Release --parallel`).

To build the `image-net` runner, use the same command you used for building ORT, (i.e. if you built in release mode, use `./build.sh --config=Release`)




 
