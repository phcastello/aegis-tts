# FastPitch checkpoint recovery

The FastPitch runner validates a locally written full checkpoint before publishing
it as `last.pt`. It saves model, AdamW, scheduler, AMP scaler, Python/NumPy/Torch/CUDA
RNG, optimizer step, epoch, data cursor, validation patience and learning-rate
reduction state. A resumed run still requires the same scientific configuration
and data manifests. The resource-change exception remains explicit in the CLI.

Publication writes and fsyncs `last.pt.tmp`, hashes and reloads that same open file
on CPU, checks finite values and state structure, and compares every saved value
with the in-memory snapshot. It then writes and fsyncs
`checkpoint.pending.json` before replacing `last.pt`. Directory fsyncs make the
journal, checkpoint, optional `best.pt` hardlink and final `checkpoint.json`
receipt durable in order. The journal is removed only after publication finishes.
The recipe, optimizer equations, validation rules and synthesis path are unchanged.

On resume, the receipt or pending journal must identify the exact checkpoint SHA.
The loader hashes before deserialization and again afterward using the same file
descriptor, while checking inode and file metadata. It validates the complete
state before restoring model, optimizer, scheduler, RNG or data position. A
committed checkpoint with a matching pending journal can finish publication after
an interruption. A temporary file or journal that has not been committed causes
the runner to stop before further updates. Preserve those files and the previous
receipt until their hashes, counters and ownership have been reconciled; repeatedly
restarting the trainer does not repair an incomplete publication.

These are trusted local checkpoints: `weights_only=False` is needed for the saved
RNG state. Hash checks establish file stability and recorded provenance; they do
not make an untrusted pickle safe to load. Atomic replacement requires files on
the same filesystem and enough room for the previous checkpoint and the temporary
replacement. External retention may keep additional hardlinked generations.

The focused tests use synthetic corruption and interrupted-publication cases,
including a learning-rate reduction. Two optional tests use a tiny CPU model to
compare the next update after resume with uninterrupted training and exercise both
package and direct-script import contexts:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B -m unittest discover -s tests -p test_v2_fastpitch_checkpoint_io.py -v
```

Torch and NumPy are required for the two real CPU cases; otherwise they are
reported as skipped. The tests load no public model, dataset or production service.
Interruption fixtures simulate failures between filesystem operations; they are
not a physical power-loss test. The local A2000 migration also completed a real
update from step 20003 to 20004, then another update to 20005 in a fresh process,
with complete CPU audits of both checkpoints. Those engineering states were
retired after qualification; the public training resumed from the original 20003
state with explicit source-migration provenance.
