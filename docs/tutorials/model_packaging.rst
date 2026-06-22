Tutorial: Model Packaging After Benchmarking / Optimization
===========================================================

After benchmarking or optimization PathBench now writes a self-contained model
package next to the best Lightning checkpoint. The package already contains the
validated config, the architecture selection, the feature-selection metadata,
and the trained weights required for direct inference or later export.

Where Checkpoints Are Saved
----------------------------

PathBench writes checkpoints to the experiment directory:

.. code-block:: text

   project_root/{project_name}/
   └── checkpoints/
       ├── {epoch}-{val_loss:.2f}.ckpt
       └── {epoch}-{val_loss:.2f}_package.pt   ← self-contained inference package

For an optimization run the best trial checkpoint is saved in the same
``checkpoints/`` directory under the optimization project:

.. code-block:: text

   project_root/{project_name}/
   └── checkpoints/
       └── {epoch}-{val_loss:.2f}.ckpt

Loading the Packaged Model
--------------------------

The packaged artifact is a regular ``torch.save`` payload with the exact model
weights plus the config required to rebuild the architecture:

.. code-block:: python

   import torch
   from pathbench.inference.model_package import load_packaged_model, predict_bag

   loaded = load_packaged_model(
       "/experiments/luad_benchmark/checkpoints/epoch=09-val_loss=0.32_package.pt"
   )
   bag = torch.randn(1, 512, loaded.package_payload["input_dim"])
   logits = predict_bag(loaded.model, bag, task=loaded.task)
   print(logits.shape)

The serialized payload contains:

- ``config``: full validated PathBench config dump
- ``model_name``: registry key used to rebuild the model
- ``input_dim`` and ``output_dim``: bag feature dimension and output channels
- ``inference_metadata``: default ``bag_id`` and ``feature_extractor`` values
- ``state_dict``: trained weights ready for loading

Exporting to TorchScript
--------------------------

For deployment without a PathBench dependency, trace the underlying
:class:`~pathbench.core.models.mil_base.MILModelBase`:

.. code-block:: python

   import torch

   # loaded.model is a MILModelBase reconstructed from the package
   model = loaded.model
   example = torch.randn(1, 512, 2048)  # [B, N, D]
   scripted = torch.jit.trace(model, example)
   torch.jit.save(scripted, "model_scripted.pt")

   # Load and run without PathBench
   scripted_model = torch.jit.load("model_scripted.pt")
   with torch.no_grad():
       logits = scripted_model(example)

Inspecting Packaged Metadata
----------------------------

.. code-block:: python

   import torch

   package = torch.load("best_model_package.pt", map_location="cpu", weights_only=False)
   print(package["model_name"])
   print(package["inference_metadata"])

Running Inference from a Bundle
--------------------------------

.. code-block:: bash

   pathbench-infer \
     --model_path /exports/luad_abmil_bundle/model_package.pt \
     --input /data/artifacts/test/SLIDE_001.h5 \
     --output /data/predictions/SLIDE_001.json

Converting to ONNX
------------------

For cross-framework deployment:

.. code-block:: python

   import torch

   # model is a MILModelBase loaded as shown above
   dummy = torch.randn(1, 512, 2048)

   torch.onnx.export(
       model,
       dummy,
       "model.onnx",
       input_names=["bag"],
       output_names=["logits"],
       dynamic_axes={
           "bag": {0: "batch_size", 1: "num_tiles"},
           "logits": {0: "batch_size"},
       },
       opset_version=17,
   )

Best Practices
--------------

- Prefer the packaged ``*_package.pt`` artifact for inference and sharing.
- Keep the original ``.ckpt`` only for Lightning-specific resume workflows.
- Verify that the packaged ``bag_id`` and ``feature_extractor`` match the
  feature artifacts you want to score.
- Test the packaged model on held-out slides before deploying.
- For survival models, check that the risk score direction matches your
  convention (higher = worse prognosis vs. higher = better prognosis).
