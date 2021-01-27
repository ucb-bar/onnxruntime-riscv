# Bumping to upstream

Since this is forked from the upstream [ONNX Runtime](https://github.com/microsoft/onnxruntime) repo, you may want to manually bump the base commit to match upstream. In most cases a simple 

```
git pull https://github.com/microsoft/onnxruntime master
```

will suffice (or you can `pull --rebase` if you're so inclined), but there are some subtle points to keep in mind.

* There will inevitably be some merge conflicts (hopefully few). While most of the systolic specific code is isolated in its own folders, there are a few places were we have to modify the core onnxruntime files in order to add providers or otherwise interop. Resolve these as appropriate; you may want to reference the upstream commit id from which the conflict originated to see why the upstream change was made. In some cases (API change) you will have to patch other files to get things to build again.

* Be careful with submodules. Maybe it's just be but git submodule management is horrible and for some reason git pull never actually bumps the submodule commits. Simplest way I found is to just manually compare the submodules (in `cmake/external`) with upstream and bump manually as needed.
 

Another important note is that sometimes you'll have to edit the build scripts for the imagenetrunner/etc. because they link to a new module. You can get the list of modules to link against by running the main ORT build with `export VERBOSE=1` so that cmake prints out the actual gcc commands. VERY IMPORTANT is that you preserve both the existence and order the wonkier libraries that we link against: the whole `-ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive` stuff. This is needed because we are statically linking the binary and glibc HATES it. So we make it happy with this one weird trick. For more info see the comment in the main ORT `build.sh` file. Without it, the binary will just segfault (in qemu it silently exits).
