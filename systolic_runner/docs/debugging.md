# Debugging Tips

* Run things in qemu using `-x 0` (Gemmini CPU mode) when possible
* For segfaults, qemu does not show you the faulting location. If you're lazy, then `printf` will work if you have intuition as to where the cause of the fault lies. Otherwise, you can run using `spike pk` and it will print out the program counter at the time of fault. You can then do `riscv64-unknown-linux-gnu-objdump -C -l -d ort_test  --start-address=0xF00F > dumped.txt` and look at the log (also maybe set ending address so you don't end up with a gigantic log). If you compile in debug mode, then you'll have the source file and line number conveniently provided. (Note that it's technically possible to use qemu userspace with gdb if you install multiarch gdb with support for riscv on your host machine but that seems way more work than is worth it. qemu does support some neat strace though https://ariadne.space/2021/05/05/using-qemu-user-emulation-to-reverse-engineer-binaries/)
* For debugging models, use Netron
* For debugging kernels, printing out entire tensors can be unwieldly. You can either print min/max of a tensor (and if you see NaNs that's a good indication of OOB read or uninitialized values), or print to a file and open in numpy. (fwrite + np.fromfile)

