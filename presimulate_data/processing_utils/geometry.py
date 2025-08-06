import numpy as np
import random

def load_strands(strands_path, num_strands):
    """
    Load strands from an obj file.
    Args:
        strands_path: Path to the file containing the strands.
    Returns:
        strands: A list of N strands, each strand is [strand_len, 3]
        
    """
    print('Loading strands from', strands_path)

    with open(strands_path, 'r') as f:
        lines = f.readlines()
    vertices = []
    strands = {}
    strand_ids = {}
    for line in lines:
        if line.startswith('v '):
            x, y, z = line.split()[1:4]
            vertices.append([float(x), float(y), float(z)])
        elif line.startswith('l '):
            vertex_ids = [int(v) for v in line.split()[1:]]
            vertex_values = [vertices[i - 1] for i in vertex_ids]
            if vertex_ids[0] > vertex_ids[-1]:
                vertex_ids = vertex_ids[::-1]
                vertex_values = vertex_values[::-1]
            if vertex_ids[0] not in strands:
                strands[vertex_ids[-1]] = vertex_values
                strand_ids[vertex_ids[-1]] = vertex_ids
            else:
                strands[vertex_ids[0]] += vertex_values[1:]
                strands[vertex_ids[-1]] = strands[vertex_ids[0]]
                strand_ids[vertex_ids[0]] += vertex_ids[1:]
                strand_ids[vertex_ids[-1]] = strand_ids[vertex_ids[0]]
                del strands[vertex_ids[0]]
                del strand_ids[vertex_ids[0]]

    ids = sorted(strand_ids.keys(), reverse=True)   
    keys_to_delete = []
    for key in ids:
        if strand_ids[key][0] in strands:
            strands[strand_ids[key][0]] += strands[key][1:]
            strand_ids[strand_ids[key][0]] += strand_ids[key][1:]
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del strands[key]
        del strand_ids[key]
            
    out_strands_x = []
    out_strands_y = []
    out_strands_z = []
    for strand in strands.values():
        out_strand_x = np.array(strand)[:, 0]
        out_strands_x.append(out_strand_x)
        out_strand_y = np.array(strand)[:, 1]
        out_strands_y.append(out_strand_y)
        out_strand_z = np.array(strand)[:, 2]
        out_strands_z.append(out_strand_z)
    x_strands = [np.array(strand) for strand in out_strands_x]
    y_strands = [np.array(strand) for strand in out_strands_y]
    z_strands = [np.array(strand) for strand in out_strands_z]

    # Choose a random subset of strands
    n = num_strands  # Number of random elements to choose
    print('Number of strands:', n)
    if len(strands) < n:
        n = len(strands)
    if n > 0:
        x_strands = random.sample(x_strands, n)
        y_strands = random.sample(y_strands, n)
        z_strands = random.sample(z_strands, n)
    
    # print average sement size
    x_segment_sizes = [] 
    y_segment_sizes = []
    z_segment_sizes = []
    for strand in x_strands:
        x_segment_sizes += [np.linalg.norm(strand[i] - strand[i+1]) for i in range(len(strand) - 1)]
    for strand in y_strands:
        y_segment_sizes += [np.linalg.norm(strand[i] - strand[i+1]) for i in range(len(strand) - 1)]
    for strand in z_strands:
        z_segment_sizes += [np.linalg.norm(strand[i] - strand[i+1]) for i in range(len(strand) - 1)]
    print('Average segment size:', np.mean(x_segment_sizes), np.mean(y_segment_sizes), np.mean(z_segment_sizes))

    print('Loaded', len(x_strands), 'strands')
    return x_strands, y_strands, z_strands

def save_strands_to_obj(strands, save_path, intermediate=False):
    """
    Save the strands to an obj file.
    Args:
        strands: A list of 3D points representing the strands.
        save_path: Path to save the strands.
    """
    curr_vertex_id = 1
    with open(save_path, 'w') as f:
        strand_edges = []
        for strand in strands:
            strand_edge = []
            for i, vertex in enumerate(strand):
                f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
                strand_edge.append(i)
            strand_edges.append(strand_edge)
        for strand_edge in strand_edges:
            f.write('l ' + ' '.join([str(curr_vertex_id + i) for i in strand_edge]) + '\n')
            curr_vertex_id += len(strand_edge)
    if not intermediate:
        print('Strands saved to', save_path)