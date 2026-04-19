"""Game Asset Variation Factory — GLB texture variation core.

Extracts embedded albedo textures from a GLB, lets callers substitute new
image bytes, and writes out a modified GLB with updated bufferView offsets.
"""

import logging
from pathlib import Path

from pygltflib import GLTF2

logger = logging.getLogger(__name__)

PRESETS = {
    "Rusted": "rusted and oxidized",
    "Damaged": "worn, scratched, and damaged",
    "Overgrown": "overgrown with moss and vines",
    "Scorched": "scorched, burned, and blackened",
    "Clean": "pristine and polished",
}

INTENSITIES = {
    "Subtle": "slightly",
    "Moderate": "noticeably",
    "Heavy": "heavily, dramatically",
}

NEGATIVE_PROMPT = "seams, artifacts, text, watermark"


def build_prompt(preset: str, intensity: str, user_text: str = "") -> str:
    desc = PRESETS[preset]
    word = INTENSITIES[intensity]
    extra = user_text.strip()
    suffix = f", {extra}" if extra else ""
    return (
        f"A tileable material texture, {word} {desc}{suffix}. "
        "Preserve original UV layout and material structure."
    )


def snap_to_valid_size(w: int, h: int) -> tuple[int, int]:
    """Square, clamped to [256, 2048], rounded down to multiple of 16."""
    target = max(w, h)
    target = max(256, min(2048, target))
    target = (target // 16) * 16
    return target, target


def sanitize_glb(input_path: str | Path, output_path: str | Path) -> Path:
    """Strip glTF extensions so minimal viewers (e.g. Gradio's Model3D) can load it.

    Clears `extensionsUsed` / `extensionsRequired` at the document level and
    any per-object `extensions` dicts on materials, textures, samplers, images,
    meshes, primitives, and nodes. Base color textures are kept intact — only
    the extension metadata is removed.
    """
    logger.info("sanitize_glb: %s -> %s", input_path, output_path)
    gltf = GLTF2().load(str(input_path))

    stripped_used = list(gltf.extensionsUsed or [])
    stripped_required = list(gltf.extensionsRequired or [])
    gltf.extensionsUsed = []
    gltf.extensionsRequired = []

    def clear(objs):
        for o in objs or []:
            if getattr(o, "extensions", None):
                o.extensions = {}
            if getattr(o, "extras", None):
                o.extras = {}

    clear(gltf.materials)
    clear(gltf.textures)
    clear(gltf.samplers)
    clear(gltf.images)
    clear(gltf.nodes)
    clear(gltf.accessors)
    clear(gltf.bufferViews)
    clear(gltf.buffers)
    clear(gltf.scenes)
    for mesh in gltf.meshes or []:
        if getattr(mesh, "extensions", None):
            mesh.extensions = {}
        clear(getattr(mesh, "primitives", None))

    output_path = Path(output_path)
    gltf.save(str(output_path))
    logger.info(
        "sanitized: removed extensionsUsed=%s extensionsRequired=%s, wrote %d bytes",
        stripped_used, stripped_required, output_path.stat().st_size,
    )
    return output_path


def _albedo_image_indices(gltf: GLTF2) -> list[int]:
    """Return sorted list of image indices used as baseColor textures."""
    indices = set()
    for mat in gltf.materials:
        pbr = mat.pbrMetallicRoughness
        if pbr is None or pbr.baseColorTexture is None:
            continue
        tex_idx = pbr.baseColorTexture.index
        if tex_idx is None or tex_idx >= len(gltf.textures):
            continue
        img_idx = gltf.textures[tex_idx].source
        if img_idx is not None:
            indices.add(img_idx)
    return sorted(indices)


def extract_albedo_textures(glb_path: str | Path) -> list[tuple[int, bytes, str]]:
    """Return [(image_index, raw_bytes, mime_type), ...] for each albedo texture.

    Raises ValueError if the GLB has no albedo texture or the texture is
    external (stored by URI instead of bufferView).
    """
    logger.info("extract_albedo_textures: loading %s", glb_path)
    gltf = GLTF2().load(str(glb_path))
    blob = gltf.binary_blob() or b""
    indices = _albedo_image_indices(gltf)
    logger.debug("albedo image indices: %s (total images in glb: %d)", indices, len(gltf.images))
    if not indices:
        raise ValueError("No albedo (baseColor) texture found in GLB.")

    results = []
    for img_idx in indices:
        img = gltf.images[img_idx]
        if img.bufferView is None:
            raise ValueError(
                f"Image {img_idx} is stored by URI; only embedded textures are supported."
            )
        bv = gltf.bufferViews[img.bufferView]
        start = bv.byteOffset or 0
        end = start + bv.byteLength
        results.append((img_idx, bytes(blob[start:end]), img.mimeType or "image/png"))
    logger.info(
        "extracted %d albedo texture(s): sizes=%s mimes=%s",
        len(results),
        [len(b) for _, b, _ in results],
        [m for _, _, m in results],
    )
    return results


def replace_albedo_textures(
    glb_path: str | Path,
    replacements: dict[int, bytes],
    output_path: str | Path,
    new_mime: str = "image/png",
) -> Path:
    """Write a copy of `glb_path` with the given image replacements applied.

    `replacements` maps image_index -> new image bytes. Output path is returned.
    Handles arbitrary size changes by rebuilding the binary blob with
    recomputed bufferView offsets.
    """
    logger.info(
        "replace_albedo_textures: %s -> %s, replacing %d image(s)",
        glb_path, output_path, len(replacements),
    )
    gltf = GLTF2().load(str(glb_path))
    blob = gltf.binary_blob() or b""

    image_to_bv = {i: img.bufferView for i, img in enumerate(gltf.images) if img.bufferView is not None}
    bv_replacements: dict[int, bytes] = {}
    for img_idx, new_bytes in replacements.items():
        bv_idx = image_to_bv.get(img_idx)
        if bv_idx is None:
            raise ValueError(f"Image {img_idx} is not stored in a bufferView.")
        bv_replacements[bv_idx] = new_bytes
        old_len = gltf.bufferViews[bv_idx].byteLength
        logger.debug("  image[%d] bv[%d]: %d -> %d bytes", img_idx, bv_idx, old_len, len(new_bytes))

    # Rebuild the binary blob. Walk bufferViews in original byteOffset order so
    # that any accessor references that depend on stable ordering keep working.
    order = sorted(
        range(len(gltf.bufferViews)),
        key=lambda i: gltf.bufferViews[i].byteOffset or 0,
    )

    parts: list[bytes] = []
    cursor = 0
    new_offsets: dict[int, tuple[int, int]] = {}

    for bv_idx in order:
        bv = gltf.bufferViews[bv_idx]
        if cursor % 4 != 0:
            pad = 4 - (cursor % 4)
            parts.append(b"\x00" * pad)
            cursor += pad

        if bv_idx in bv_replacements:
            data = bv_replacements[bv_idx]
        else:
            start = bv.byteOffset or 0
            data = bytes(blob[start : start + bv.byteLength])

        new_offsets[bv_idx] = (cursor, len(data))
        parts.append(data)
        cursor += len(data)

    if cursor % 4 != 0:
        pad = 4 - (cursor % 4)
        parts.append(b"\x00" * pad)
        cursor += pad

    new_blob = b"".join(parts)

    for bv_idx, (offset, length) in new_offsets.items():
        gltf.bufferViews[bv_idx].byteOffset = offset
        gltf.bufferViews[bv_idx].byteLength = length

    for img_idx in replacements:
        gltf.images[img_idx].mimeType = new_mime

    if gltf.buffers:
        gltf.buffers[0].byteLength = len(new_blob)

    gltf.set_binary_blob(new_blob)
    output_path = Path(output_path)
    gltf.save(str(output_path))
    logger.info("wrote %s (%d bytes, blob=%d bytes)", output_path, output_path.stat().st_size, len(new_blob))
    return output_path
