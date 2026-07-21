import os
import numpy as np


def _load_covis_graph(scene_root: str, scene_meta: dict):
    sm = scene_meta.get("scene_modalities", {})

    def _load_csr_mmap_npy(dir_path: str):
        indptr = np.load(os.path.join(dir_path, "indptr.npy"), mmap_mode="r")
        indices = np.load(os.path.join(dir_path, "indices.npy"), mmap_mode="r")
        data = np.load(os.path.join(dir_path, "data.npy"), mmap_mode="r")
        shape = np.load(os.path.join(dir_path, "shape.npy"))
        shape = tuple(shape.tolist())
        return {
            "format": "csr",
            "indptr": indptr,
            "indices": indices,
            "data": data,
            "shape": shape,
        }

    key_view = sm.get("covis_graph_view_csr")
    if key_view is None:
        raise KeyError("scene_meta.scene_modalities missing 'covis_graph_view_csr'")

    rel = key_view.get("scene_key", None)
    fmt = key_view.get("format", "")

    if rel is None:
        raise KeyError("covis_graph_view_csr missing 'scene_key'")

    abs_path = os.path.join(scene_root, rel)

    if fmt in ("csr_mmap_npy", "csr_mmap") or os.path.isdir(abs_path):
        return _load_csr_mmap_npy(abs_path)

    raise FileNotFoundError(f"Cannot load covis graph: format={fmt}, path={abs_path}")


def _is_csr_graph(x):
    """Check if input is a CSR graph dictionary."""
    return isinstance(x, dict) and x.get("format", None) in ["csr", "csr_npz"]


def _csr_row(g, i: int):
    """Return neighbors and weights for row i from CSR graph."""
    indptr = g["indptr"]
    indices = g["indices"]
    data = g["data"]
    s = int(indptr[i])
    e = int(indptr[i + 1])
    return indices[s:e], data[s:e]


def _keep_w_in_range(
    w: np.ndarray,
    w_min: float = None,
    w_max: float = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return boolean mask where w in [w_min, w_max)."""
    keep = np.ones_like(w, dtype=bool)
    if w_min is not None:
        keep &= (w >= float(w_min))
    if w_max is not None:
        keep &= (w < float(w_max) - eps)
    return keep


def _csr_edge(g, i: int, j: int) -> float:
    """Return edge weight from i to j, 0 if edge doesn't exist."""
    nbrs, w = _csr_row(g, i)
    if nbrs.size == 0:
        return 0.0
    for n, ww in zip(nbrs, w):
        if int(n) == int(j):
            return float(ww)
    return 0.0


def _edge_w(g, u: int, v: int, bidirectional: bool = True) -> float:
    """Return edge weight between u and v (optionally bidirectional)."""
    w = float(_csr_edge(g, u, v))
    if bidirectional:
        w = max(w, float(_csr_edge(g, v, u)))
    return w


def _weighted_choice(nbrs, w, temperature=1.0, rng: np.random.Generator = None):
    """Sample one neighbor proportional to softmax(w / temperature)."""
    if len(nbrs) == 0:
        raise ValueError("Empty candidates in _weighted_choice.")
    if len(nbrs) == 1:
        return int(nbrs[0])

    if rng is None:
        rng = np.random.default_rng()

    w = np.asarray(w, dtype=np.float32)
    t = max(float(temperature), 1e-8)
    z = (w / t) - (w / t).max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)

    return int(rng.choice(nbrs, p=p))


def _weighted_sample_without_replacement(
    nbrs,
    w,
    k: int,
    temperature=1.0,
    rng: np.random.Generator = None,
):
    """Sample k unique neighbors proportional to softmax(w / temperature)."""
    if rng is None:
        rng = np.random.default_rng()

    nbrs = np.asarray(nbrs)
    w = np.asarray(w, dtype=np.float32)

    if nbrs.size == 0 or k <= 0:
        return np.array([], dtype=np.int64)

    k = min(int(k), int(nbrs.size))
    if k == int(nbrs.size):
        return nbrs.astype(np.int64, copy=False)

    t = max(float(temperature), 1e-8)
    z = (w / t) - (w / t).max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)

    chosen = rng.choice(nbrs, size=k, replace=False, p=p)
    return np.asarray(chosen, dtype=np.int64)


def _get_effective_neighbors(
    g,
    cur: int,
    visited=None,
    min_covis: float = None,
    max_covis: float = None,
    topk_step: int = None,
    bidirectional_edge: bool = True,
):
    """
    Return filtered neighbors and effective weights for one node.
    """
    nbrs, w_dir = _csr_row(g, cur)
    if nbrs.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    if bidirectional_edge:
        w_eff = np.empty_like(w_dir, dtype=np.float32)
        for k, (n, wd) in enumerate(zip(nbrs, w_dir)):
            n = int(n)
            w_eff[k] = max(float(wd), float(_csr_edge(g, n, cur)))
    else:
        w_eff = w_dir.astype(np.float32, copy=False)

    keep = _keep_w_in_range(w_eff, min_covis, max_covis)
    nbrs = nbrs[keep]
    w_eff = w_eff[keep]
    if nbrs.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    if visited is not None:
        keep = np.array([int(n) not in visited for n in nbrs], dtype=bool)
        nbrs = nbrs[keep]
        w_eff = w_eff[keep]
        if nbrs.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    if topk_step is not None and nbrs.size > int(topk_step):
        k = int(topk_step)
        part = np.argpartition(w_eff, -k)[-k:]
        nbrs = nbrs[part]
        w_eff = w_eff[part]

    return nbrs.astype(np.int64, copy=False), w_eff.astype(np.float32, copy=False)


def _anchor_star_sampling_csr(
    g,
    num_of_samples: int,
    min_covis: float,
    max_covis: float,
    topk_step: int,
    temperature: float,
    max_retries: int,
    bidirectional_edge: bool,
    rng: np.random.Generator,
):
    """
    Anchor-star sampling:
    1) choose one anchor node
    2) randomly sample the remaining nodes from anchor's overlapping neighbors
    """
    N = int(g["shape"][0])
    if N <= 0:
        return np.array([], dtype=np.int64)

    best_walk = np.array([], dtype=np.int64)
    candidates = list(range(N))

    for _ in range(max_retries):
        anchor = int(rng.choice(candidates))
        nbrs, w = _get_effective_neighbors(
            g,
            anchor,
            visited={anchor},
            min_covis=min_covis,
            max_covis=max_covis,
            topk_step=topk_step,
            bidirectional_edge=bidirectional_edge,
        )

        walk = [anchor]
        if nbrs.size > 0:
            picked = _weighted_sample_without_replacement(
                nbrs,
                w,
                k=num_of_samples - 1,
                temperature=temperature,
                rng=rng,
            )
            walk.extend([int(x) for x in picked])

        walk = np.asarray(walk, dtype=np.int64)
        if len(walk) > len(best_walk):
            best_walk = walk
        if len(walk) >= num_of_samples:
            return walk

    return best_walk


def _random_walk_sampling_csr(
    g,
    num_of_samples: int,
    max_retries: int,
    min_covis: float,
    max_covis: float,
    restart_prob: float,
    temperature: float,
    topk_step: int,
    bidirectional_edge: bool,
    rng: np.random.Generator,
):
    """
    Random walk sampling on CSR graph.
    """
    N = int(g["shape"][0])
    if N <= 0:
        return np.array([], dtype=np.int64)

    excluded_nodes = set()
    best_walk = []

    for _ in range(max_retries):
        visited = set()
        walk = []
        stack = []

        all_nodes = set(range(N))
        available_nodes = list(all_nodes - excluded_nodes)
        if not available_nodes:
            break

        start = int(rng.choice(available_nodes))
        walk.append(start)
        visited.add(start)
        stack.append(start)

        while len(walk) < num_of_samples and stack:
            cur_main = stack[-1]
            use_start = rng.random() < restart_prob
            cur = start if use_start else cur_main

            nbrs, w = _get_effective_neighbors(
                g,
                cur,
                visited=visited,
                min_covis=min_covis,
                max_covis=max_covis,
                topk_step=topk_step,
                bidirectional_edge=bidirectional_edge,
            )

            # if restart from start fails, fallback to current stack top
            if nbrs.size == 0 and cur != cur_main:
                nbrs, w = _get_effective_neighbors(
                    g,
                    cur_main,
                    visited=visited,
                    min_covis=min_covis,
                    max_covis=max_covis,
                    topk_step=topk_step,
                    bidirectional_edge=bidirectional_edge,
                )
                cur = cur_main

            if nbrs.size == 0:
                stack.pop()
                continue

            nxt = _weighted_choice(
                nbrs,
                w,
                temperature=temperature,
                rng=rng,
            )
            walk.append(nxt)
            visited.add(nxt)
            stack.append(nxt)

        if len(walk) > len(best_walk):
            best_walk = walk
        if len(walk) >= num_of_samples:
            return np.array(walk, dtype=np.int64)

        excluded_nodes.update(visited)

    return np.array(best_walk, dtype=np.int64)


def _greedy_chain_sampling_csr_once(
    g,
    num_of_samples: int,
    min_covis: float,
    max_covis: float,
    topk_step: int,
    start: int,
    bidirectional_edge: bool,
    enforce_global_max: bool,
    rng: np.random.Generator,
):
    """
    Greedy chain sampling: at each step choose highest-weight neighbor.
    """
    N = int(g["shape"][0])
    if N <= 0:
        return np.array([], dtype=np.int64)

    if start is None:
        start = int(rng.integers(0, N))

    eps = 1e-12
    walk = [start]
    visited = {start}
    cur = start

    while len(walk) < num_of_samples:
        nbrs, w_eff = _get_effective_neighbors(
            g,
            cur,
            visited=None,  # greedy needs manual visited/global checks below
            min_covis=min_covis,
            max_covis=max_covis,
            topk_step=topk_step,
            bidirectional_edge=bidirectional_edge,
        )
        if nbrs.size == 0:
            break

        order = np.argsort(-w_eff)
        chosen = None

        for j in order:
            cand = int(nbrs[j])
            if cand in visited:
                continue

            if enforce_global_max and (max_covis is not None) and (len(walk) >= 2):
                ok = True
                for u in walk:
                    if u == cand:
                        ok = False
                        break
                    if _edge_w(g, int(u), cand, bidirectional=bidirectional_edge) >= float(max_covis) - eps:
                        ok = False
                        break
                if not ok:
                    continue

            chosen = cand
            break

        if chosen is None:
            break

        walk.append(chosen)
        visited.add(chosen)
        cur = chosen

    return np.array(walk, dtype=np.int64)


def _greedy_chain_sampling_csr(
    g,
    num_of_samples: int,
    min_covis: float,
    max_covis: float,
    topk_step: int,
    max_retries: int,
    bidirectional_edge: bool,
    rng: np.random.Generator,
):
    """
    Greedy chain sampling with multiple retries.
    """
    N = int(g["shape"][0])
    best_walk = np.array([], dtype=np.int64)

    candidates = list(range(N))
    if not candidates:
        return best_walk

    for _ in range(max_retries):
        start = int(rng.choice(candidates))
        walk = _greedy_chain_sampling_csr_once(
            g,
            num_of_samples,
            min_covis=min_covis,
            max_covis=max_covis,
            topk_step=topk_step,
            start=start,
            bidirectional_edge=bidirectional_edge,
            enforce_global_max=True,
            rng=rng,
        )
        if len(walk) > len(best_walk):
            best_walk = walk
        if len(walk) >= num_of_samples:
            return walk

    return best_walk


def _tree_sampling_csr(
    g,
    num_of_samples: int,
    min_covis: float,
    max_covis: float,
    topk_step: int,
    temperature: float,
    max_retries: int,
    bidirectional_edge: bool,
    tree_branching: int,
    tree_trunk_ratio: float,
    rng: np.random.Generator,
):
    """
    Tree sampling:
    1) build a short trunk from a root, using greedy-chain-like selection
    2) expand children from trunk / visited nodes in BFS-like order

    trunk:
        - choose highest-weight valid neighbor at each step
        - similar to greedy_chain
    branch:
        - still use weighted random sampling for diversity
    """
    N = int(g["shape"][0])
    if N <= 0:
        return np.array([], dtype=np.int64)

    best_walk = np.array([], dtype=np.int64)
    candidates = list(range(N))
    eps = 1e-12

    for _ in range(max_retries):
        root = int(rng.choice(candidates))
        visited = {root}
        walk = [root]

        # ---------- step 1: build trunk (greedy-chain-like) ----------
        trunk_len = max(2, int(np.ceil(num_of_samples * float(tree_trunk_ratio))))
        trunk_len = min(trunk_len, num_of_samples)

        cur = root
        while len(walk) < trunk_len:
            nbrs, w_eff = _get_effective_neighbors(
                g,
                cur,
                visited=None,  # trunk 自己手动做 visited / global max 约束
                min_covis=min_covis,
                max_covis=max_covis,
                topk_step=topk_step,
                bidirectional_edge=bidirectional_edge,
            )
            if nbrs.size == 0:
                break

            order = np.argsort(-w_eff)
            chosen = None

            for j in order:
                cand = int(nbrs[j])

                # 1) 不能重复
                if cand in visited:
                    continue

                # 2) 类似 greedy_chain，加一个全局 max_covis 去冗余约束
                if (max_covis is not None) and (len(walk) >= 2):
                    ok = True
                    for u in walk:
                        if u == cand:
                            ok = False
                            break
                        if _edge_w(g, int(u), cand, bidirectional=bidirectional_edge) >= float(max_covis) - eps:
                            ok = False
                            break
                    if not ok:
                        continue

                chosen = cand
                break

            if chosen is None:
                break

            walk.append(chosen)
            visited.add(chosen)
            cur = chosen

        # ---------- step 2: expand branches ----------
        # 以 trunk + 后续加入节点为父节点队列，BFS 风格向外长分支
        queue = list(walk)
        qi = 0

        while len(walk) < num_of_samples and qi < len(queue):
            parent = int(queue[qi])
            qi += 1

            nbrs, w = _get_effective_neighbors(
                g,
                parent,
                visited=visited,
                min_covis=min_covis,
                max_covis=max_covis,
                topk_step=topk_step,
                bidirectional_edge=bidirectional_edge,
            )
            if nbrs.size == 0:
                continue

            num_child = min(
                int(tree_branching),
                num_of_samples - len(walk),
                len(nbrs),
            )

            picked = _weighted_sample_without_replacement(
                nbrs,
                w,
                k=num_child,
                temperature=temperature,
                rng=rng,
            )

            for c in picked:
                c = int(c)
                if c in visited:
                    continue
                walk.append(c)
                visited.add(c)
                queue.append(c)
                if len(walk) >= num_of_samples:
                    break

        walk = np.asarray(walk, dtype=np.int64)

        if len(walk) > len(best_walk):
            best_walk = walk
        if len(walk) >= num_of_samples:
            return walk

    return best_walk

def _sample_mode_from_mixture(
    rng: np.random.Generator,
    mixed_anchor_star_prob: float,
    mixed_random_walk_prob: float,
    mixed_tree_prob: float,
    mixed_greedy_chain_prob: float,
):
    modes = np.array(
        ["anchor_star", "random_walk", "tree", "greedy_chain"],
        dtype=object,
    )
    probs = np.array(
        [
            mixed_anchor_star_prob,
            mixed_random_walk_prob,
            mixed_tree_prob,
            mixed_greedy_chain_prob,
        ],
        dtype=np.float64,
    )
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0:
        probs = np.array([0.5, 0.25, 0.15, 0.10], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(modes, p=probs))


def _csr_sampling(
    view_graph,
    num_of_samples,
    rng: np.random.Generator,
    max_retries=4,
    sampling_mode="random_walk",   # "anchor_star" | "random_walk" | "tree" | "greedy_chain" | "mixed"
    use_bidirectional_covis=True,
    covisibility_thres=0.05,
    covisibility_thres_max=1.0,
    topk_step=50,
    walk_restart_prob=0.10,
    walk_temperature=1.0,
    tree_branching=2,
    tree_trunk_ratio=0.25,
    mixed_anchor_star_prob=0.50,
    mixed_random_walk_prob=0.25,
    mixed_tree_prob=0.15,
    mixed_greedy_chain_prob=0.10,
):
    """
    Sampling on CSR view graph.
    """
    mode = sampling_mode
    if mode == "mixed":
        mode = _sample_mode_from_mixture(
            rng=rng,
            mixed_anchor_star_prob=mixed_anchor_star_prob,
            mixed_random_walk_prob=mixed_random_walk_prob,
            mixed_tree_prob=mixed_tree_prob,
            mixed_greedy_chain_prob=mixed_greedy_chain_prob,
        )

    if mode == "anchor_star":
        return _anchor_star_sampling_csr(
            view_graph,
            num_of_samples,
            min_covis=covisibility_thres,
            max_covis=covisibility_thres_max,
            topk_step=topk_step,
            temperature=walk_temperature,
            max_retries=max_retries,
            bidirectional_edge=use_bidirectional_covis,
            rng=rng,
        )

    if mode == "greedy_chain":
        return _greedy_chain_sampling_csr(
            view_graph,
            num_of_samples,
            min_covis=covisibility_thres,
            max_covis=covisibility_thres_max,
            topk_step=topk_step,
            max_retries=max_retries,
            bidirectional_edge=use_bidirectional_covis,
            rng=rng,
        )

    if mode == "tree":
        return _tree_sampling_csr(
            view_graph,
            num_of_samples,
            min_covis=covisibility_thres,
            max_covis=covisibility_thres_max,
            topk_step=topk_step,
            temperature=walk_temperature,
            max_retries=max_retries,
            bidirectional_edge=use_bidirectional_covis,
            tree_branching=tree_branching,
            tree_trunk_ratio=tree_trunk_ratio,
            rng=rng,
        )

    # default = random_walk
    return _random_walk_sampling_csr(
        view_graph,
        num_of_samples,
        max_retries=max_retries,
        min_covis=covisibility_thres,
        max_covis=covisibility_thres_max,
        restart_prob=walk_restart_prob,
        temperature=walk_temperature,
        topk_step=topk_step,
        bidirectional_edge=use_bidirectional_covis,
        rng=rng,
    )
