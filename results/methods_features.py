import matplotlib.pyplot as plt
import numpy as np

from fad.features import filters as sf


def get_center(length):
    if (length % 2):
        c = (length-1)/2
    else:
        c = length/2
    return int(c)


# Stats
INTER_PUPILLARY_DISTANCE = 32
WIDTH = INTER_PUPILLARY_DISTANCE * 4
HEIGHT = round(WIDTH*1.5)
face_width_per_image_width = INTER_PUPILLARY_DISTANCE*2/WIDTH
face_width_per_image_height = INTER_PUPILLARY_DISTANCE*2/HEIGHT
cc = get_center(WIDTH)
cr = get_center(HEIGHT)
shape  = (HEIGHT, WIDTH)
results = sf.face_filters(shape, INTER_PUPILLARY_DISTANCE)
ffilters = results["ffilters"] # (row, col, sf, angle)
sfilters = results["sfilters"] # (row, col, sf, angle)
_, _, nscale, norient = ffilters.shape
# sf    --> {lowest, middle, highest}
# angle --> {__, vertical, __, __, horizontal, __}


print("\n\n")
print("Vertical frequencies")
print(f'\tlow\tcenter\thigh\toctaves')
for s in range(nscale):
    fd = np.fft.fftshift(ffilters[:,:,s,1])
    transfer = np.copy( fd[cr,cc:] )

    # fig, ax = plt.subplots()
    # ax.plot(transfer, 'o-', linewidth=2, markeredgewidth=2)
    # ax.set(xscale="log")
    # plt.show()

    fc = np.argmax(transfer)
    fl = np.argmin(abs(transfer[:fc-1] - .5))
    transfer[:fc-1] = transfer.max()
    fh = np.argmin(abs(transfer - .5))
    bandwidth_octaves = (fh / fl) / 2
    fl *= face_width_per_image_width
    fc *= face_width_per_image_width
    fh *= face_width_per_image_width
    print(f'\t{fl:.1f}\t{fc:.1f}\t{fh:.1f}\t{bandwidth_octaves}')
print(f'\tFilter separation\t{fh/fc:.2f} multiplier')
print(f'\tFilter separation\t{(fh/fc)/2:.2f} octaves')

print("\n\n")
print("Horizontal frequencies")
print(f'\tlow\tcenter\thigh\toctaves')
for s in range(nscale):
    fd = np.fft.fftshift(ffilters[:,:,s,4])
    transfer = np.copy( fd[:cr,cc] )
    transfer = np.flip(transfer)

    # fig, ax = plt.subplots()
    # ax.plot(transfer, 'o-', linewidth=2, markeredgewidth=2)
    # ax.set(xscale="log")
    # plt.show()

    fc = np.argmax(transfer)
    fl = np.argmin(abs(transfer[:fc-1] - .5))
    transfer[:fc-1] = transfer.max()
    fh = np.argmin(abs(transfer - .5))
    bandwidth_octaves = (fh / fl) / 2
    fl *= face_width_per_image_height
    fc *= face_width_per_image_height
    fh *= face_width_per_image_height
    print(f'\t{fl:.1f}\t{fc:.1f}\t{fh:.1f}\t{bandwidth_octaves}')
print(f'\tFilter separation\t{fh/fc:.2f} multiplier')
print(f'\tFilter separation\t{(fh/fc)/2:.2f} octaves')


# Plot all
X, Y = sf.grid_normalised(HEIGHT, WIDTH)
fig, ax = plt.subplots(nrows=1, ncols=2)
orient = [o for o in range(norient)]

C = ["blue","red","green"]
for s in range(nscale):
    for o in orient[::2]:
        fd = np.fft.fftshift(ffilters[:,:,s,o])
        ax[0].contour(X, Y, fd, levels=np.array([.5]), colors=C[s])
        ax[0].contour(X, Y, np.flip(fd,axis=(0,1)), levels=np.array([.5]), colors=C[s])
for s in range(nscale):
    for o in orient[1::2]:
        fd = np.fft.fftshift(ffilters[:,:,s,o])
        ax[1].contour(X, Y, fd, levels=np.array([.5]), colors=C[s])
        ax[1].contour(X, Y, np.flip(fd,axis=(0,1)), levels=np.array([.5]), colors=C[s])
ax[0].set_aspect("equal")
ax[1].set_aspect("equal")
ax[0].tick_params(left = False, right = False , labelleft = False , 
                  labelbottom = False, bottom = False)
ax[1].tick_params(left = False, right = False , labelleft = False , 
                  labelbottom = False, bottom = False)
plt.show()
fig.savefig("method-wrff-wavelets-sf.pdf")


# End
# -------------------------------------------------------------------