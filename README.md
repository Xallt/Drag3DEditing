# Drag 3D Editing


## Common issues

### npx: Permission denied
```bash
(viser) No client build found. Building now...
(viser) nodejs is set up!
bash: line 1: /workspace/Drag3DEditing/deps/GaussianEditor/extern/viser/src/viser/client/.nodeenv/bin/npx: Permission denied
bash: line 1: /workspace/Drag3DEditing/deps/GaussianEditor/extern/viser/src/viser/client/.nodeenv/bin/npx: Permission denied
```

A bug in Viser [on this line](https://github.com/nerfstudio-project/viser/blob/main/src/viser/_client_autobuild.py#L60) leads to the `npm` binaries not having execution rights

Fix:
```bash
chmod -R +x /workspace/Drag3DEditing/deps/GaussianEditor/extern/viser/src/viser/client/.nodeenv
```

replace with your path to the created `.nodeenv`