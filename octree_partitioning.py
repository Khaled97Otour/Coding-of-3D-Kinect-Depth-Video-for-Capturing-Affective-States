import numpy as np

CORNERS = np.array([
    [0,0,0], [1,0,0], [0,1,0], [1,1,0],
    [0,0,1], [1,0,1], [0,1,1], [1,1,1]
], dtype=np.uint8)

def compute_all_bboxes(bbox_min, bbox_max):
    midpoint = (bbox_max - bbox_min) // 2 + bbox_min
    bbox_mins = np.where(CORNERS==1, midpoint, bbox_min)
    bbox_maxs = np.where(CORNERS==1, bbox_max, midpoint)
    local_bboxes = [(np.zeros(3, dtype=np.int32), bbox_maxs[i]-bbox_mins[i]) for i in range(8)]
    return bbox_mins, bbox_maxs, local_bboxes

def compute_new_bbox(child_index, bbox_min, bbox_max):
    midpoint = (bbox_max - bbox_min) // 2 + bbox_min
    bits = [(child_index>>i)&1 for i in range(3)]
    new_min = np.array([midpoint[i] if bits[i] else bbox_min[i] for i in range(3)], dtype=np.int32)
    new_max = np.array([bbox_max[i] if bits[i] else midpoint[i] for i in range(3)], dtype=np.int32)
    return new_min, new_max

def split_octree(points, bbox_min, bbox_max):
    midpoint = (bbox_max - bbox_min) // 2 + bbox_min
    loc = ((points[:,0]>=midpoint[0]).astype(np.uint8) |
           ((points[:,1]>=midpoint[1])<<1) |
           ((points[:,2]>=midpoint[2])<<2))

    # Vectorized split without Python loop
    ret_points = [points[loc==i] for i in range(8)]
    nonempty_idx = np.where([len(rp)>0 for rp in ret_points])[0]
    ret_points_nonempty = [ret_points[i] for i in nonempty_idx]
    binstr = np.uint8(np.bitwise_or.reduce(1<<nonempty_idx)) if len(nonempty_idx)>0 else np.uint8(0)

    bbox_mins, bbox_maxs, _ = compute_all_bboxes(bbox_min, bbox_max)
    return ret_points_nonempty, binstr, bbox_mins, bbox_maxs

def partition_octree_rec(points, bbox_min, bbox_max, level):
    if points.shape[0]==0 or level==0:
        return [points], []

    blocks, binstr, bbox_mins, bbox_maxs = split_octree(points, bbox_min, bbox_max)
    all_blocks, all_binstrs = [], []

    for i in range(len(blocks)):
        sub_blocks, sub_binstrs = partition_octree_rec(blocks[i], bbox_mins[i], bbox_maxs[i], level-1)
        all_blocks.extend(sub_blocks)
        all_binstrs.extend(sub_binstrs)

    all_binstrs = [binstr]+all_binstrs
    return all_blocks, all_binstrs

def partition_octree(points, bbox_min, bbox_max, level):
    bbox_min, bbox_max = np.asarray(bbox_min,dtype=np.int32), np.asarray(bbox_max,dtype=np.int32)
    points = np.asarray(points,dtype=np.int32)
    return partition_octree_rec(points, bbox_min, bbox_max, level)

def departition_octree(blocks, binstr_list, bbox_min, bbox_max, level):
    bbox_min, bbox_max = np.asarray(bbox_min,dtype=np.int32), np.asarray(bbox_max,dtype=np.int32)
    blocks, binstr_list = blocks.copy(), binstr_list.copy()
    block_idx, cur_level = 0, 1
    bbox_stack, parents_stack = [(bbox_min,bbox_max)], []

    while block_idx < len(blocks):
        child_found = False
        binstr = binstr_list[0]
        child_indices = [i for i in range(8) if (binstr & (1<<i))]

        while child_indices and not child_found:
            v = child_indices.pop(0)
            cur_bbox = compute_new_bbox(v, *bbox_stack[-1])
            if cur_level==level:
                blocks[block_idx] = blocks[block_idx]
                block_idx += 1
            else:
                child_found = True
            binstr_list[0] &= ~(1<<v)

        if child_found:
            bbox_stack.append(cur_bbox)
            parents_stack.append(0)
            cur_level += 1
        else:
            if not parents_stack: break
            _ = parents_stack.pop()
            cur_level -= 1
            bbox_stack.pop()

    return blocks

