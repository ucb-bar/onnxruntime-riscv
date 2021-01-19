# Bumping to upstream

Since this is forked from the upstream [ONNX Runtime](https://github.com/microsoft/onnxruntime) repo, you may want to manually bump the base commit to match upstream. In most cases a simple 

```
git pull https://github.com/microsoft/onnxruntime master
```

will suffice (or you can `pull --rebase` if you're so inclined), but there are some subtle points to keep in mind.

* There will inevitably be some merge conflicts (hopefully few). While most of the systolic specific code is isolated in its own folders, there are a few places were we have to modify the core onnxruntime files in order to add providers or otherwise interop. Resolve these as appropriate; you may want to reference the upstream commit id from which the conflict originated to see why the upstream change was made. In some cases (API change) you will have to patch other files to get things to build again.

* Be careful with submodules. Maybe it's just be but git submodule management is horrible and for some reason git pull never actually bumps the submodule commits. Simplest way I found is to just manually compare the submodules (in `cmake/external`) with upstream and bump manually as needed.
 
