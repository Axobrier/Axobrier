#!/usr/bin/env python3
# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import json
import random
from typing import List, Dict, Tuple

# Set random seed for reproducible datasets
random.seed(42)

# ═══════════════════════════════════════════════════════════════════
#  DOMAIN 1: MESH & PBR RULES (cartridge_mesh_pbr)
# ═══════════════════════════════════════════════════════════════════

MESH_PBR_CLASSES = {
    "MAX_4_BONE_INFLUENCES": [
        "Vertex assigned 5 bone weights in vertex group panel. Rigging exceeds the target engine limit of 4 influences.",
        "Error: vertex group count for vertex 4829 exceeds 4 bone influences. Mesh deformation requires weight pruning.",
        "Skinning artifact: excessive bone count detected on clavicle vertex. Clamp bone influences to 4 per vertex.",
        "Mesh validation failed: 124 vertices have more than 4 bone weights assigned. Apply limit total 4 in Blender.",
        "Vertex weights exceed 4 per vertex limit in avatar armature. Pruning weights below 0.05 threshold.",
        "Rigging error: vertex group table has 6 bone influences on elbow joint. Engine will discard lowest weights.",
        "Fitted mesh check: bone weights per vertex > 4 detected on skirt rig. Normalize and limit total to 4.",
        "Avatar upload rejection: vertex has 5 bone weights. real-time mesh skinning requires max 4 influences.",
        "Shoulder joint deformation artifact: vertex is weighted to 5 bones simultaneously in avatar rig.",
        "Character deformation artifact: vertex weighted to multiple bones exceeding the maximum 4 influences.",
        "Mesh skinning error: joint deformation artifact caused by more than 4 bone weights on vertex.",
        "Avatar rigging inspection: bone weights per vertex exceed limit; prune lowest influences."
    ],
    "ORM_TEXTURE_PACKING": [
        "Texture packing: combine ambient occlusion, roughness, and metallic into a single RGB texture image.",
        "Material channel setup: Occlusion mapped to Red channel, Roughness to Green channel, Metallic to Blue channel.",
        "Packing glTF ORM map: Red = AO, Green = Roughness, Blue = Metallic. Decouple base color from surface properties.",
        "PBR metallic-roughness workflow requires packed ORM texture. Slot occlusion in R, roughness in G, metallic in B.",
        "Bake texture setup: combine separate AO, Roughness, and Metalness passes into composite RGB ORM bitmap.",
        "Shading pipeline error: roughness texture placed in Red channel instead of Green. Packing requires R=AO, G=Roughness, B=Metal.",
        "Shader setup: pack ambient occlusion into R, perceptual roughness into G, metallic mask into B for glTF export.",
        "PBR material audit: unpack ORM composite map to inspect individual ambient occlusion and roughness channels."
    ],
    "ALPHA_MODE_MASK_FOR_HAIR": [
        "Alpha sorting artifact: transparent hair cards flickering in viewport. Change material alpha mode from BLEND to MASK.",
        "Hair strands render in wrong depth order due to alpha blending. Set alpha cutoff threshold to 0.5 with MASK mode.",
        "Z-buffer sorting glitch on layered transparency. Use Alpha Clip or Alpha Mask instead of Alpha Blend on foliage and hair.",
        "Viewer depth sorting failure on overlapping hair geometry. Switch material alpha mode to MASK with 0.5 cutoff.",
        "Alpha fighting on eyelashes and hair mesh. Avoid Alpha Blending; configure glTF material alphaMode to MASK.",
        "Visual artifact: hair cards showing background through front strands. Enable Alpha Masking to write to depth buffer.",
        "Transparent geometry sorting error. Hair cards must use alphaMode MASK to ensure proper rasterization depth order.",
        "Hair mesh alpha failure: use dual-pass alpha masking or strict 1-bit clip threshold to prevent transparency sorting artifacts."
    ],
    "DAE_ARMATURE_RIGID_EXPORT": [
        "COLLADA export configuration for avatar mesh: enable engine compatible matrix transforms and apply unit scale.",
        "Blender DAE export setup: check OpenCOLLADA format, include armatures, and apply rotation and scale.",
        "Rigged mesh import error: joints deformed on upload. Re-export COLLADA .dae with engine avatar compatible preset.",
        "Export settings: use COLLADA (.dae), uncheck include animations, check include armature, and ensure Z-up coordinate system.",
        "Avatar clothing export: export selected objects as COLLADA .dae with bind shape matrix aligned to avatar skeleton.",
        "COLLADA validation error: root bone transform matrix non-standard. Export with avatar compatibility flags enabled.",
        "Mesh attachment import: DAE file missing bind pose or joint hierarchy. Verify OpenCOLLADA export profile.",
        "Real-time engine rigged export: export .dae with transformation matrix applied and armature hierarchy preserved."
    ],
    "GLTF_PBR_SEPARATE_NORMAL": [
        "Normal map configuration: tangent space normal map must be authored in separate RGB texture using MikkTSpace.",
        "glTF 2.0 PBR standard: normal map is decoupled from ORM packed texture and assigned to its own texture sampler.",
        "Shader setup: normal texture must use non-color data space and OpenGL Y+ tangent normal orientation.",
        "Normal map baking: bake tangent space normals using standard MikkTSpace tangents; do not pack into ORM image.",
        "PBR material export: separate normal map in tangent space format with unit length normal vectors.",
        "glTF material validator: normalTexture property references independent texture index with standard scale factor.",
        "Surface bump details: normal map assigned to separate sampler slot with linear color space interpretation.",
        "Tangent space normal map error: DirectX Y- inverted normal map detected. Invert green channel for OpenGL/glTF standard."
    ],
    "CONVEX_HULL_PHYSICS_DECOMP": [
        "Physics mesh optimization: decompose non-convex collision mesh into maximum 256 convex hulls.",
        "High physics cost on upload. Replace concave triangle mesh with volumetric convex hull decomposition.",
        "Havok collision engine warning: non-convex geometry in physics shape. Run convex hull decomposition tool in Blender.",
        "Physics shape analysis: collision model contains degenerate holes and self-intersections. Decompose into simple convex polyhedra.",
        "Collision geometry error: physics shape has too many vertices. Simplify collision hull to reduce land impact weight.",
        "Collision mesh setup: build bounding boxes and convex decomposition hulls to minimize physical calculation overhead.",
        "Physics cost penalty: triangle list physics shape exceeds limit. Switch physics type to Convex Hull in mesh upload.",
        "Physics hull decomposition: generate watertight convex collision hulls without internal faces or concave pockets."
    ],
    "LOD_4_REDUCTION_STRATEGY": [
        "Land impact optimization: generate 4 manual levels of detail (High, Medium, Low, Lowest) to minimize download weight.",
        "Mesh LOD strategy: Lowest LOD reduced to single quad billboard or 1 triangle for small props to lower streaming cost.",
        "Upload weight penalty: high land impact caused by unoptimized Low and Lowest LOD meshes. Decimate geometry progressively.",
        "Level of detail reduction: High LOD 10,000 tris, Medium LOD 2,500 tris, Low LOD 500 tris, Lowest LOD 20 tris.",
        "Optimize streaming performance: author custom simplified meshes for all 4 LOD tiers instead of using auto-decimation.",
        "LOD decimation artifact: silhouette collapses at distance. Manually preserve outline in Low LOD mesh generation.",
        "Download weight reduction: Lowest LOD triangle count must be kept below 50 triangles to prevent extreme land impact.",
        "4-tier LOD hierarchy: bake normal maps from High LOD onto Medium and Low LOD meshes to retain visual fidelity."
    ],
    "LIMIT_65K_VERTICES_PER_SURFACE": [
        "Mesh import error: surface 0 contains 72,000 vertices, exceeding the 65,536 (16-bit index) vertex buffer limit.",
        "Engine geometry limit: split mesh into multiple material slots because single material exceeds 65k vertex ceiling.",
        "Index buffer overflow: triangle indices exceed 65,535 limit. Subdivide object into separate material sub-meshes.",
        "Mesh validation failed: 81,920 vertices in single material group. Partition model into multiple draw calls.",
        "Vertex count exceeded: 16-bit hardware index buffer cannot address more than 65,536 vertices per draw surface.",
        "Material partitioning required: high-poly sculpt exceeds 65k vertex threshold. Split mesh across multiple materials.",
        "Engine vertex buffer crash: object has 95,000 vertices on single material. Separate mesh into sub-sections under 65k.",
        "Geometry export error: draw call limit exceeded. Keep each material surface strictly under 65,536 vertex indices."
    ]
}

# ═══════════════════════════════════════════════════════════════════
#  DOMAIN 2: RUNTIME PERFORMANCE TRIAGE (cartridge_runtime_perf)
# ═══════════════════════════════════════════════════════════════════

PERF_TRIAGE_CLASSES = {
    "GC_ALLOC_IN_UPDATE_LOOP": [
        "Frame rate micro-stutter detected: 12KB garbage collection allocation occurring every frame inside OnUpdate().",
        "Performance profile warning: LINQ query allocation inside tight per-frame render tick triggers frequent GC pauses.",
        "Memory allocation in hot loop: instantiating new StringBuilder() every frame causing 50MB/sec garbage generation.",
        "Frame time spike: GC.Collect() invoked due to boxing allocations inside physics FixedUpdate callback.",
        "Allocation anti-pattern: lambda closure allocating delegates every tick. Cache delegate instance in member variable.",
        "Profiling trace: List.ToArray() called inside animation update loop causes 3ms garbage collection pauses every 2 seconds.",
        "High GC pressure: temporary byte array allocated per frame in network serialization loop. Use pooled byte buffer.",
        "Frame stuttering diagnostic: memory allocation inside Update() loop triggers gen 0 GC collections every 100 frames."
    ],
    "UNINDEXED_DATABASE_QUERY": [
        "Slow query alert: sequential table scan on accounts table taking 4,200ms. Missing B-tree index on organization_id.",
        "Database bottleneck: SELECT query filter on created_at and status performs full scan across 12 million rows.",
        "Execution plan warning: table scan detected on users table. Add composite index on (tenant_id, email).",
        "PostgreSQL diagnostic: Seq Scan on orders table filter (customer_id = $1). Missing index on foreign key column.",
        "High CPU utilization on database: unindexed LIKE prefix query forces full table scan on 5 million transactions.",
        "Database query timeout: missing index on join column leads to hash join fallback with 15-second response latency.",
        "Query optimization report: query cost 45,000 on payments table. Create partial index on status WHERE status = 'PENDING'.",
        "Performance degradation: unindexed sorting query on timestamps forces disk-based filesort across entire table."
    ],
    "BLOCKING_IO_ON_EVENT_LOOP": [
        "Thread pool starvation: synchronous File.ReadAllText() called directly on UI event loop thread blocks rendering for 80ms.",
        "Node.js event loop lag: fs.readFileSync() inside HTTP request handler blocks all concurrent connections for 250ms.",
        "Latency spike detected: synchronous database socket read executed on main reactor thread halts event processing.",
        "Event loop blocked: synchronous crypto hashing operation on main worker thread delays I/O event polling by 500ms.",
        "Thread freeze: synchronous socket connect() on main dispatcher thread causes UI window to stop responding.",
        "Event loop latency 320ms: synchronous disk log writing on high-frequency request path blocks worker thread.",
        "Anti-pattern detected: blocking HTTP GET call executed inside async event callback without task offloading.",
        "Reactor thread stall: synchronous file lock acquisition blocks all reactive pipeline streams."
    ],
    "UNBOUNDED_THREAD_SPAWNING": [
        "Resource exhaustion: new Thread().start() invoked per incoming socket connection leads to 4,000 OS threads and OOM.",
        "Thread starvation: creating unmanaged threads for background tasks causes OS context switching overhead to exceed 40%.",
        "System crash: OutOfMemoryError: unable to create new native thread. Use thread pool or worker semaphore instead.",
        "High concurrency failure: 8,000 raw threads spawned under load, exhausting OS thread stack memory limits.",
        "Thread leak detected: naked threads spawned in loop without termination condition or join mechanism.",
        "Operating system thrashing: excessive active threads cause kernel scheduler contention and CPU saturation.",
        "Performance alert: thread count reached 2,500. Replace per-request thread spawning with bounded ExecutorService.",
        "Process failure: thread creation rate outpaces thread completion, exhausting virtual address space for thread stacks."
    ],
    "HOT_PATH_STRING_CONCAT": [
        "Performance bottleneck: string concatenation (+) inside 100,000-iteration loop allocates 400MB of intermediate strings.",
        "High CPU and memory churn: str += chunk inside file parser creates O(N^2) memory reallocation. Use StringBuilder.",
        "Trace diagnostic: String.format() called inside tight packet decoding loop accounts for 35% of total request execution time.",
        "Hot path allocation: repetitive string concatenation in logging path causes excessive string object instantiations.",
        "Optimization required: replace string plus operator with pre-sized character buffer in JSON serialization routine.",
        "Micro-benchmark failure: string concatenation in loop is 80x slower than pre-allocated array join.",
        "Memory profiler alert: 70% of heap retained by ephemeral string instances generated by + operators in data pipeline.",
        "Inefficient text formatting: string interpolation inside per-packet parsing loop degrades throughput by 4x."
    ],
    "N_PLUS_ONE_ORM_FETCH": [
        "Database query explosion: loading 50 orders generates 51 individual SQL SELECT queries due to lazy loading.",
        "ORM anti-pattern detected: accessing user.profile inside loop triggers N+1 database round-trips for each iteration.",
        "High latency in API endpoint: 150 database queries executed for single page view. Add JOIN FETCH or select_related.",
        "SQL log audit: repeated SELECT * FROM line_items WHERE order_id = ? executed 200 times in single web request.",
        "Database connection exhaustion: N+1 lazy loading queries overwhelm connection pool under concurrent user traffic.",
        "Performance review: Hibernate lazy initialization in serializer causes 80 distinct round-trips to SQL server.",
        "Query cascade warning: fetching parent entity without eager join executes hundreds of child queries in nested loop.",
        "API response time 2.4s: 90% of duration spent in sequential database round-trips caused by N+1 ORM navigation."
    ],
    "LEAKED_EVENT_SUBSCRIPTION": [
        "Memory leak detected: transient view controller subscribes to global notification center without unsubscribing on destroy.",
        "Heap growth diagnostic: 50,000 dead UI objects retained in memory because singleton event dispatcher holds strong references.",
        "Object lifecycle bug: event listener registered on root event bus retains discarded widget instance indefinitely.",
        "Retained memory leak: missing removeEventListener on component unmount causes detached DOM tree to remain in memory.",
        "Memory profiler alert: ServiceManager instance holds strong reference delegates to closed session objects.",
        "Gradual memory creep: client session objects never garbage collected due to dangling event handler registrations.",
        "Event subscription leak: WeakReference or explicit unbind missing in messaging bus listener registration.",
        "Heap snapshot analysis: 1.2GB memory retained by event listener array holding references to destroyed views."
    ],
    "UNBOUNDED_MEMORY_CACHE": [
        "Process OOM crash: in-memory dictionary cache has no eviction policy and grows indefinitely with incoming user IDs.",
        "Memory leak: global HashMap used as cache without maximum size or TTL eviction exhausts heap space after 48 hours.",
        "Resource warning: cache size exceeded 10 million entries without LRU eviction. Replace with bounded Guava / Caffeine cache.",
        "Unbounded cache growth: memoized function storing all input arguments without expiration causes gradual memory degradation.",
        "Heap exhaustion: response cache grows unbounded under DDoS attack with randomized query parameters.",
        "Memory alert: in-memory token cache lacks size cap or soft references, triggering OutOfMemoryError after 3 days.",
        "Cache anti-pattern: storing large binary payloads in unbounded ConcurrentHashMap without eviction threshold.",
        "Diagnostic trace: heap memory steadily climbs 50MB/hour due to unbounded static lookup table retaining stale keys."
    ]
}


def generate_dataset_variants(
    base_examples: List[str],
    label: str,
    target_count: int = 100
) -> List[Dict[str, str]]:
    """
    Expands base seed examples into diverse training variations using template
    augmentation, prefixing, and subtle noise.
    """
    prefixes = [
        "Diagnostic log: ",
        "Code review finding: ",
        "Automated rule check: ",
        "Runtime telemetry warning: ",
        "Profiler incident report: ",
        "Static analysis failure: ",
        "Architecture guideline violation: ",
        "Performance alert: ",
        "Asset inspection error: ",
        "Compiler warning: ",
        "Production incident: ",
        ""
    ]
    suffixes = [
        " Please investigate.",
        " High severity issue.",
        " Requires immediate remediation.",
        " Rule violation confirmed.",
        " Fails validation gate.",
        " Fix required before merge.",
        " Priority P1 triage.",
        ""
    ]

    results = []
    # 1. Add all base examples
    for ex in base_examples:
        results.append({"text": ex, "label": label})

    # 2. Augment until target_count reached
    while len(results) < target_count:
        base = random.choice(base_examples)
        pref = random.choice(prefixes)
        suff = random.choice(suffixes)

        # Apply minor stylistic variations
        if random.random() < 0.3:
            var_text = f"{pref}{base.lower()}{suff}"
        elif random.random() < 0.6:
            var_text = f"{pref}{base}{suff}"
        else:
            words = base.split()
            if len(words) > 5 and random.random() < 0.4:
                # Slight word truncation / slice
                var_text = f"{pref}{' '.join(words[:len(words)-2])}.{suff}"
            else:
                var_text = f"{pref}{base}{suff}"

        results.append({"text": var_text.strip(), "label": label})

    return results


def build_domain_dataset(
    domain_classes: Dict[str, List[str]],
    samples_per_class: int = 100
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Builds train (80%) and validation (20%) datasets for a domain.
    """
    train_data = []
    val_data = []

    for label, seeds in domain_classes.items():
        all_samples = generate_dataset_variants(seeds, label, samples_per_class)
        random.shuffle(all_samples)

        split_idx = int(len(all_samples) * 0.8)
        train_data.extend(all_samples[:split_idx])
        val_data.extend(all_samples[split_idx:])

    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data


def save_jsonl(data: List[Dict[str, str]], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(data)} records to {filepath}")


def main():
    print("===========================================================")
    print("  Axobrier - Synthetic Domain Dataset Generator")
    print("===========================================================\n")

    out_dir = os.path.join(os.path.dirname(__file__), "data")

    # 1. Generate Domain 1: Mesh & PBR Rules
    print("Generating dataset: Mesh & PBR Rules (8 classes)...")
    mesh_train, mesh_val = build_domain_dataset(MESH_PBR_CLASSES, samples_per_class=120)
    save_jsonl(mesh_train, os.path.join(out_dir, "mesh_pbr_train.jsonl"))
    save_jsonl(mesh_val, os.path.join(out_dir, "mesh_pbr_val.jsonl"))
    print(f"  Classes: {list(MESH_PBR_CLASSES.keys())}\n")

    # 2. Generate Domain 2: Runtime Performance Triage
    print("Generating dataset: Runtime Performance Triage (8 classes)...")
    perf_train, perf_val = build_domain_dataset(PERF_TRIAGE_CLASSES, samples_per_class=120)
    save_jsonl(perf_train, os.path.join(out_dir, "runtime_perf_train.jsonl"))
    save_jsonl(perf_val, os.path.join(out_dir, "runtime_perf_val.jsonl"))
    print(f"  Classes: {list(PERF_TRIAGE_CLASSES.keys())}\n")

    print("All domain datasets successfully generated in training/data/")


if __name__ == "__main__":
    main()
