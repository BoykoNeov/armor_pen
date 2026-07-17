extends RefCounted
class_name CacheLoader
##
## Parses a cache directory (manifest.json + frames.bin) per docs/CACHE_FORMAT.md
## and streams frames on demand. Knows ONLY the cache format — no solver, no
## physics (CLAUDE.md §2).
##
## Column layout is read from the manifest at load time; offsets are NEVER
## hardcoded (CACHE_FORMAT §2). Add an attribute to the solver's output and this
## loader keeps working — it just exposes one more named column.

## v2 (milestone 13) appended `internal_energy`. Nothing below changed to read it:
## the loader locates columns by name, so it needed only this number. That is the
## §2 openness rule paying out — but the gate still moves, because a v1 cache does
## not have the column and a reader must be able to tell before it looks.
const SUPPORTED_SCHEMA_VERSION := 2

var particle_count: int = 0
var frame_count: int = 0
var attributes: PackedStringArray = PackedStringArray()
var domain: Dictionary = {}
var materials: Dictionary = {}
var frame_dt: float = 0.0

var _stride: int = 0                 # floats per particle record
var _frame_bytes: int = 0            # bytes per full frame
var _bin_path: String = ""
var _attr_index: Dictionary = {}     # attribute name -> column index


## Load a cache directory. Returns OK, or an error string describing the problem.
func load_cache(dir_path: String) -> Variant:
	var manifest_path := dir_path.path_join("manifest.json")
	if not FileAccess.file_exists(manifest_path):
		return "no manifest.json in %s" % dir_path

	var text := FileAccess.get_file_as_string(manifest_path)
	var manifest = JSON.parse_string(text)
	if manifest == null or not (manifest is Dictionary):
		return "manifest.json is not valid JSON"

	if int(manifest.get("schema_version", -1)) != SUPPORTED_SCHEMA_VERSION:
		return "unsupported schema_version %s" % str(manifest.get("schema_version"))
	if String(manifest.get("dtype", "")) != "float32":
		return "unsupported dtype %s" % str(manifest.get("dtype"))

	particle_count = int(manifest["particle_count"])
	frame_count = int(manifest["frame_count"])
	attributes = PackedStringArray(manifest["attributes"])
	domain = manifest["domain"]
	materials = manifest.get("materials", {})
	frame_dt = float(manifest["frame_dt"])

	_stride = attributes.size()
	_frame_bytes = particle_count * _stride * 4    # sizeof(float32)
	_bin_path = dir_path.path_join("frames.bin")
	_attr_index.clear()
	for i in attributes.size():
		_attr_index[attributes[i]] = i

	if not FileAccess.file_exists(_bin_path):
		return "no frames.bin in %s" % dir_path
	var expected := frame_count * _frame_bytes
	var actual := int(FileAccess.open(_bin_path, FileAccess.READ).get_length())
	if actual != expected:
		return "frames.bin is %d bytes, expected %d" % [actual, expected]

	return OK


## Column index for a named attribute, or -1 if absent (viewer picks a fallback).
func attribute_index(name: String) -> int:
	return _attr_index.get(name, -1)


## Read frame `f` as a flat PackedFloat32Array of size particle_count * stride,
## laid out [p0_a0, p0_a1, ..., p1_a0, ...] (CACHE_FORMAT §3).
func read_frame(f: int) -> PackedFloat32Array:
	assert(f >= 0 and f < frame_count, "frame index out of range")
	var file := FileAccess.open(_bin_path, FileAccess.READ)
	file.seek(f * _frame_bytes)                     # direct seek — §3
	var bytes := file.get_buffer(_frame_bytes)
	file.close()
	return bytes.to_float32_array()
