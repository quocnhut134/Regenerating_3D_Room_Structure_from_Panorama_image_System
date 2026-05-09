import json
import numpy as np
import os
import shutil

def export_full_room(json_path, img_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    cam_h = data["camera_height"]
    room_h = data["layout_height"] 
    corners = data["floor_corners_3d"]
    n = len(corners)
    
    y_floor = -cam_h 
    y_ceil = room_h - cam_h 

    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, "3d_model.obj")
    mtl_path = os.path.join(output_dir, "3d_model.mtl")
    img_name = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(output_dir, img_name))

    with open(mtl_path, 'w') as f:
        f.write("newmtl mat_pano\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n")
        f.write(f"map_Kd {img_name}\n\n")
        f.write("newmtl mat_gray\nKa 0.200 0.200 0.200\nKd 0.500 0.500 0.500\n")

    vertices = []
    uvs = []
    faces_interior = []
    faces_exterior = []

    def calc_uv(x, y, z):
        d = np.sqrt(x*x + y*y + z*z)
        if d == 0: return 0.5, 0.5
        u = 0.5 + np.arctan2(x, -z) / (2 * np.pi)
        v = 0.5 + np.arcsin(y / d) / np.pi
        return u, v

    def build_surface_grid(p_tl, p_tr, p_br, p_bl, grid_res=50):
        start_idx = len(vertices) + 1
        for j in range(grid_res + 1):
            v_f = j / grid_res
            p_l = p_tl + (p_bl - p_tl) * v_f
            p_r = p_tr + (p_br - p_tr) * v_f
            prev_u = None
            for i in range(grid_res + 1):
                u_f = i / grid_res
                p = p_l + (p_r - p_l) * u_f
                u, v = calc_uv(p[0], p[1], p[2])
                if prev_u is not None:
                    if u - prev_u > 0.5: u -= 1.0
                    elif prev_u - u > 0.5: u += 1.0
                prev_u = u
                vertices.append(p)
                uvs.append((u, v))
        
        for j in range(grid_res):
            for i in range(grid_res):
                v1 = start_idx + j * (grid_res + 1) + i
                v2 = v1 + 1
                v3 = v2 + (grid_res + 1)
                v4 = v1 + (grid_res + 1)
                
                faces_interior.append((v1, v2, v3, v4))
                faces_exterior.append((v1, v4, v3, v2))

    for i in range(n):
        c1, c2 = corners[i], corners[(i+1)%n]
        build_surface_grid(
            np.array([c1['x'], y_ceil, c1['z']]), np.array([c2['x'], y_ceil, c2['z']]),
            np.array([c2['x'], y_floor, c2['z']]), np.array([c1['x'], y_floor, c1['z']])
        )

    center = np.mean([[c['x'], c['z']] for c in corners], axis=0)
    p_center_f = np.array([center[0], y_floor, center[1]])
    p_center_c = np.array([center[0], y_ceil, center[1]])

    for i in range(n):
        c1, c2 = corners[i], corners[(i+1)%n]
        build_surface_grid(
            p_center_f, p_center_f, 
            np.array([c2['x'], y_floor, c2['z']]), np.array([c1['x'], y_floor, c1['z']]), grid_res=20
        )
        build_surface_grid(
            np.array([c1['x'], y_ceil, c1['z']]), np.array([c2['x'], y_ceil, c2['z']]),
            p_center_c, p_center_c, grid_res=20
        )

    with open(obj_path, 'w') as f:
        f.write("mtllib 3d_model.mtl\n")
        
        for v in vertices: 
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        center_x = center[0]
        center_z = center[1]
        for v in vertices:
            ext_x = center_x + (v[0] - center_x) * 1.01 
            ext_z = center_z + (v[2] - center_z) * 1.01
            ext_y = v[1] * 1.01                        
            f.write(f"v {ext_x:.4f} {ext_y:.4f} {ext_z:.4f}\n")
            
        for uv in uvs: 
            f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
        
        f.write("\nusemtl mat_pano\n")
        for face in faces_interior: 
            f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]} {face[3]}/{face[3]}\n")
            
        num_v = len(vertices)
        f.write("\nusemtl mat_gray\n")
        for face in faces_exterior: 
            f.write(f"f {face[0]+num_v} {face[1]+num_v} {face[2]+num_v} {face[3]+num_v}\n")

if __name__ == '__main__':
    export_full_room("./outputs/3_layouts/AFimg0001_layout3d.json", "./outputs/3_layouts/AFimg0001.png", "./outputs/4_renders/AFimg0001_room")