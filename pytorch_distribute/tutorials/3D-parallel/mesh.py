
from torch.distributed import init_device_mesh

device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(1, 2), mesh_dim_names=('dp', 'sp'))

print('sp group', device_mesh['sp'].size(), device_mesh['sp'].get_group())
print('dp group', device_mesh['dp'].size(), device_mesh['dp'].get_group())
