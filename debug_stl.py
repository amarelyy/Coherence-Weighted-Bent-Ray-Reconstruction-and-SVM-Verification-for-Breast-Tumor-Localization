import trimesh, numpy as np

mesh = trimesh.load('data/A1.stl', force='mesh')
z_mid = (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2
print(f'z_mid = {z_mid:.2f}')

section = mesh.section(plane_origin=[0, 0, z_mid], plane_normal=[0, 0, 1])
print(f'section type: {type(section)}')
print(f'section is None: {section is None}')

if section is not None:
    attrs = [x for x in dir(section) if not x.startswith('_')]
    print(f'section attrs: {attrs}')
    
    if hasattr(section, 'vertices'):
        print(f'section.vertices shape: {section.vertices.shape}')
    
    try:
        p2d = section.to_2D()
        print(f'to_2D type: {type(p2d)}')
        attrs2 = [x for x in dir(p2d) if not x.startswith('_')]
        print(f'to_2D attrs: {attrs2}')
        if hasattr(p2d, 'vertices'):
            print(f'to_2D vertices shape: {np.array(p2d.vertices).shape}')
        if hasattr(p2d, 'entities'):
            print(f'to_2D entities: {len(p2d.entities)}')
        if hasattr(p2d, 'discrete'):
            print(f'to_2D discrete: {len(p2d.discrete)} paths')
    except Exception as e:
        print(f'to_2D error: {e}')
