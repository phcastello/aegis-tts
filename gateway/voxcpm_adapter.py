"""Optional existing DSP/onset stages around the native PCM24k backend."""
from __future__ import annotations
import asyncio
from contextlib import aclosing
from output_onset import OutputOnsetTrimmer
from voice_pipeline import AcousticStreamProcessor,Pcm16Reassembler


async def processed_stream(backend,text,cancelled,*,dsp='passthrough',onset_trim='off',initial_segment='stable'):
    if dsp not in {'passthrough','world-w04-p07-m02'}:raise ValueError('Unsupported experimental DSP')
    processor=AcousticStreamProcessor(24000,initial_segment=initial_segment) if dsp!='passthrough' else None
    reassembler=Pcm16Reassembler() if processor is not None else None
    onset=OutputOnsetTrimmer(24000) if onset_trim=='conservative-v1' else None
    async def call(function,*args):
        task=asyncio.create_task(asyncio.to_thread(function,*args))
        try:return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task;raise
    async with aclosing(backend.stream(text,cancelled)) as source:
        async for pcm in source:
            if cancelled.is_set():return
            blocks=await call(processor.push,reassembler.feed(pcm)) if processor is not None else [pcm]
            for block in blocks:
                for output in onset.push(block) if onset is not None else [block]:
                    if output and not cancelled.is_set():yield output
    if cancelled.is_set():return
    if processor is not None:
        reassembler.finish()
        for block in await call(processor.finish):
            for output in onset.push(block) if onset is not None else [block]:
                if output and not cancelled.is_set():yield output
    if onset is not None:
        for output in onset.finish():
            if output and not cancelled.is_set():yield output
