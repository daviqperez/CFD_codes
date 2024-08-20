import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define a line for printing (to ensure they have the same length)
lineSingle = '------------------------------------------------'
lineDouble = '================================================'

print (lineDouble)
print ('')
print (' solve2DDiffusionEquationPlate.py')
print ('')

print (lineDouble)

###############################################################################
#    User Inputs
###############################################################################

# plate geometry (m)
plateLength = 4
plateWidth = 4
thickness = 0.1
# propierties at the plate (deg C)
tempLeft = 100
heatfluxRight = 100
tempTop = 250
tempBot = 150
cond = 100
heatSourcePerVol = 1000
# Number of cells in the mesh
numCellsLength = 4
numCellsWidth = 4

###############################################################################
#     MESH
###############################################################################

print (' Creating Mesh')
print (lineSingle)

#calculate the number of cells
numCells = numCellsLength*numCellsWidth
# Calculate the number of nodes
numNodes = (numCellsLength+1)*(numCellsWidth+1)

# Coordinates of the cell nodes
x = np.linspace(0, plateLength, numCellsLength+1)
y = np.linspace(0, plateWidth, numCellsWidth+1)
xNodes, yNodes = np.meshgrid(x, y)

# Coordinates of the centroids
xCentroid = 0.5*(x[1:] + x[0:-1])
yCentroid = 0.5*(y[1:] + y[0:-1])
xCentroids, yCentroids = np.meshgrid(xCentroid, yCentroid)

# centroids cell distances
d_LP = 2.0*(xCentroids - xNodes[:-1,:-1]).flatten()
d_PT = 2.0*(yNodes[1:,1:] - yCentroids ).flatten()
d_PR = 2.0*(xNodes[1:,1:] - xCentroids).flatten()
d_BP = 2.0*(yCentroids- yNodes[:-1,:-1]).flatten()

# Calculate the cell volumes. 
cellLength = plateLength/numCellsLength
cellWidth = plateWidth/numCellsWidth
cellVolume = cellLength*cellWidth*thickness

# Calculate the cross sectional area in the x and y directions
areaX = cellWidth*thickness
areaY = cellLength*thickness

# Identify the cells which have boundary faces.
boundaryID = np.zeros((numCellsLength, numCellsWidth))
l_boundaryID = boundaryID.copy()
l_boundaryID[:,0] = 1
r_boundaryID = boundaryID.copy()
r_boundaryID[:,-1] = 1
t_boundaryID = boundaryID.copy()
t_boundaryID[0,:] = 1
b_boundaryID = boundaryID.copy()
b_boundaryID[-1,:] = 1

l_boundaryID = l_boundaryID.flatten()
r_boundaryID = r_boundaryID.flatten()
t_boundaryID = t_boundaryID.flatten()
b_boundaryID = b_boundaryID.flatten()

###############################################################################
#     Matrix Coefficients
###############################################################################

print (' Calculating Matrix Coefficients')
print (lineSingle)

# Diffusive flux per unit area
DA_l = np.divide(cond*areaX, d_LP)
DA_r = np.divide(cond*areaX, d_PR)
DA_b = np.divide(cond*areaY, d_BP)
DA_t = np.divide(cond*areaY, d_PT)

# Source term Su
Su = cellVolume*np.ones(numCells)*heatSourcePerVol

# Add the contribution from each of the boundary faces
Su += (2.0*tempLeft*np.multiply(l_boundaryID, DA_l))
Su += (-heatfluxRight*np.multiply(r_boundaryID, areaX)) # Su corresponding to heaflux boundary
Su += (2.0*tempBot*np.multiply(b_boundaryID, DA_b))
Su += (2.0*tempTop*np.multiply(t_boundaryID, DA_t))

# Sp interior cells
Sp = np.zeros(numCells)

# Add the contribution from each of the boundary faces
Sp += (-2.0*DA_l*l_boundaryID)
Sp += 0  # Sp corresponding to heatflux boundary 
Sp += (-2.0*DA_b*b_boundaryID)
Sp += (-2.0*DA_t*t_boundaryID)

# aL, aR, aT, aB
aL = np.multiply(DA_l, 1 - l_boundaryID)
aR = np.multiply(DA_r, 1 - r_boundaryID)
aB = np.multiply(DA_b, 1 - b_boundaryID)
aT = np.multiply(DA_t, 1 - t_boundaryID)

# Calculate ap from the other coefficients
aP = aL + aR + aB + aT - Sp

###############################################################################
#     Create the matrices
###############################################################################

print (' Assembling Matrices')
print (lineSingle)

# create the coefficients matrix and Su vector
SuVector = Su
Amatrix = np.diag(aP)

for i in range(numCells):
    # Does the cell have a left boundary?
    if (l_boundaryID[i] == 0.0):
        Amatrix[i, i-1] = -1.0*aL[i]

    # Does the cell have a right boundary?
    if (r_boundaryID[i] == 0.0):
        Amatrix[i, i+1] = -1.0*aR[i]

    # Does the cell have a bottom boundary?
    if (b_boundaryID[i] == 0.0):
        Amatrix[i, i+numCellsLength] = -1.0*aB[i]

    # Does the cell have a top boundary?
    if (t_boundaryID[i] == 0.0):
        Amatrix[i, i-numCellsLength] = -1.0*aT[i]

np.set_printoptions(linewidth=np.inf)
print (Amatrix)
print (lineSingle)

###############################################################################
#     Solve the matrices
###############################################################################

print (' Solving ...')
print (lineSingle)

Tvector = np.linalg.solve(Amatrix, SuVector)

# Reshape the vector into a grid
Tgrid = Tvector.reshape(numCellsWidth, numCellsLength)

# Print Results
print (' Solution: Temperature Field')
print (lineSingle)
print (Tgrid)
print (lineSingle)

###############################################################################
#     Interpolate temperatures node 
###############################################################################

# Interpolate the solution on the interior nodes from the CFD solution
Tleftrightnodes = 0.5*(Tgrid[:, 1:]+Tgrid[:, :-1])
Tinternalnodes = 0.5*(Tleftrightnodes[1:, :] + Tleftrightnodes[:-1, :])

# Use central differencing to calculate the Right boundary temperature in the face centroid
tempRight = (Tgrid[:,-1:] - (heatfluxRight*areaX)/(2.0*np.copy(DA_r[0])))
# calculate the Right boundary temperature in the face nodes
Trightnodes = 0.5*(tempRight[1:, :] + tempRight[:-1, :])


# Interpolate the boundary temperatures in the corners
temperatureTopLeftCorner = 0.5*(tempTop+tempLeft)
temperatureTopRightCorner = 0.5*(tempTop+tempRight[0])
temperatureBottomLeftCorner = 0.5*(tempBot+tempLeft)
temperatureBottomRightCorner = 0.5*(tempBot+tempRight[-1])

# Assemble the temperatures on all the boundary nodes
temperatureTopVector = np.hstack([temperatureTopLeftCorner,tempTop*np.ones(numCellsLength-1),temperatureTopRightCorner])
temperatureBottomVector = np.hstack([temperatureBottomLeftCorner,tempBot *np.ones(numCellsLength-1),temperatureBottomRightCorner])
temperatureLeftVector = tempLeft*np.ones([numCellsWidth-1, 1])
temperatureRightVector = Trightnodes

# Assemble the temperature on all of the nodes together as one grid
Tnodes = np.vstack([temperatureTopVector, np.hstack([temperatureLeftVector,Tinternalnodes, temperatureRightVector]),temperatureBottomVector])

# X and Y coordinates of the nodes
xNodes = xNodes.reshape([numCellsWidth+1, numCellsLength+1])
yNodes = np.flipud(yNodes.reshape([numCellsWidth+1, numCellsLength+1]))

###############################################################################
#     Check solution
###############################################################################

# print (' Check solution')
# print (lineSingle)

# # Stack on the boundary temperatures onto the grid
# Tleftrightshift = np.hstack([tempLeft*np.ones([numCellsWidth, 1]),
#                             Tgrid,
#                             tempRight*np.ones([numCellsWidth, 1])])
# Ttopbottomshift = np.vstack([tempTop*np.ones([numCellsLength]),
#                             Tgrid,
#                             tempBot*np.ones([numCellsLength])])

# # Calculate the temperature differences
# deltaTleft = Tleftrightshift[:, 1:-1]-Tleftrightshift[:, 0:-2]
# deltaTright = Tleftrightshift[:, 2:]-Tleftrightshift[:, 1:-1]
# deltaTtop = Ttopbottomshift[0:-2, :]-Ttopbottomshift[1:-1, :]
# deltaTbottom = Ttopbottomshift[1:-1, :] - Ttopbottomshift[2:, :]

# # reshape the DA vectors into a grid of the correct size
# DA_left_grid = DA_l.reshape(numCellsWidth, numCellsLength)
# DA_right_grid = DA_r.reshape(numCellsWidth, numCellsLength)
# DA_top_grid = DA_t.reshape(numCellsWidth, numCellsLength)
# DA_bottom_grid = DA_b.reshape(numCellsWidth, numCellsLength)

# # Calculate the boundary face fluxes
# DA_left_boundary = ((2.0*cond*areaX/d_LP[0]) *
#                     np.ones([numCellsWidth, 1]))
# DA_right_boundary = ((2.0*cond*areaX/d_PR[0]) *
#                      np.ones([numCellsWidth, 1]))
# DA_top_boundary = ((2.0*cond*areaY/d_PT[0]) *
#                    np.ones([numCellsLength]))
# DA_bottom_boundary = ((2.0*cond*areaY/d_BP[0]) *
#                       np.ones([numCellsLength]))

# # Stack on the boundary face fluxes to the grid
# DA_left_shift = np.hstack([DA_left_boundary, DA_left_grid[:, 1:]])
# DA_right_shift = np.hstack([DA_right_grid[:, 0:-1], DA_right_boundary])
# DA_top_shift = np.vstack([DA_top_boundary, DA_top_grid[1:, :]])
# DA_bottom_shift = np.vstack([DA_bottom_grid[0:-1, :], DA_top_boundary])

# # Unit normal vectors
# normalsLeftGrid = -1.0*np.ones([numCellsWidth, numCellsLength])
# normalsRightGrid = np.ones([numCellsWidth, numCellsLength])
# normalsBottomGrid = -1.0*np.ones([numCellsWidth, numCellsLength])
# normalsTopGrid = np.ones([numCellsWidth, numCellsLength])

# # Compute the heat fluxes
# heatFluxLeft = -np.multiply(np.multiply(DA_left_shift, deltaTleft),
#                             normalsLeftGrid)
# heatFluxRight = -np.multiply(np.multiply(DA_right_shift, deltaTright),
#                              normalsRightGrid)
# heatFluxTop = -np.multiply(np.multiply(DA_top_shift, deltaTtop),
#                            normalsTopGrid)
# heatFluxBottom = -np.multiply(np.multiply(DA_bottom_shift, deltaTbottom),
#                               normalsBottomGrid)

# # Calculate the volumetric heat source in each cell
# sourceVol = heatSourcePerVol*cellVolume*np.ones([numCellsWidth,
#                                                 numCellsLength])

# # Calculate the error in the heat flux balance in each cell
# error = (sourceVol - heatFluxLeft - heatFluxRight - heatFluxTop -
#          heatFluxBottom)

# # Reshape the matrices into vectors for printing
# heatFluxLeftVector = heatFluxLeft.flatten()
# heatFluxRightVector = heatFluxRight.flatten()
# heatFluxTopVector = heatFluxTop.flatten()
# heatFluxBottomVector = heatFluxBottom.flatten()
# sourceVolVector = sourceVol.flatten()
# errorVector = error.flatten()

# print (' Cell Heat Flux Balance')
# print (lineSingle)
# print (' Cell|  QL   |  QR   |  QT   |  QB   |  SV   |  Err')
# print (lineSingle)
# for i in range(numCells):
#     print (('%4i %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f' % (
#         i+1, heatFluxLeftVector[i], heatFluxRightVector[i],
#         heatFluxTopVector[i], heatFluxBottomVector[i],
#             sourceVolVector[i], errorVector[i])))
# print (lineSingle)

# # Sum the heat fluxes across the boundary faces to give the total heat flux across each boundary
# heatFluxLeftBoundaryTotal = np.sum(np.multiply(l_boundaryID,
#                                    heatFluxLeftVector))
# heatFluxRightBoundaryTotal = np.sum(np.multiply(r_boundaryID,
#                                     heatFluxRightVector))
# heatFluxBottomBoundaryTotal = np.sum(np.multiply(b_boundaryID,
#                                      heatFluxBottomVector))
# heatFluxTopBoundaryTotal = np.sum(np.multiply(t_boundaryID,
#                                   heatFluxTopVector))
# heatFluxBoundaryTotal = (heatFluxLeftBoundaryTotal
#                          + heatFluxRightBoundaryTotal
#                          + heatFluxTopBoundaryTotal
#                          + heatFluxBottomBoundaryTotal)
# heatGeneratedTotal = np.sum(sourceVolVector)


# print (' Boundary Heat Flux Balance')
# print (lineSingle)
# print ((' Left      : %7.1f [W]' % (heatFluxLeftBoundaryTotal)))
# print ((' Right     : %7.1f [W]' % (heatFluxRightBoundaryTotal)))
# print ((' Bottom    : %7.1f [W]' % (heatFluxBottomBoundaryTotal)))
# print ((' Top       : %7.1f [W]' % (heatFluxTopBoundaryTotal)))
# print ((' Total     : %7.1f [W]' % (heatFluxBoundaryTotal)))
# print ((' Generated : %7.1f [W]' % (heatGeneratedTotal)))
# print ((' Error     : %7.1f [W]' % (heatFluxBoundaryTotal -
#                                     heatGeneratedTotal)))
# print (lineSingle)

###############################################################################
#     Plot the solution
###############################################################################

fontSize = 11
fontSizeLegend = 11
lineWidth = 1.5
tickPad = 8
tickPad2 = 16
labelPadY = 3
labelPadX = 2
boxPad = 2
tickLength = 4
markerSize = 4

# Colour
darkBlue = '#002147'

# Use latex 'CMU sans-serif' font in the plots.
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = lineWidth
plt.rcParams["figure.figsize"] = (3.1, 2.5)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
fig1.tight_layout(pad=boxPad)
cmap_reversed = cm.get_cmap('autumn_r')
CS = ax.contourf(xNodes, yNodes, Tnodes, cmap=cmap_reversed)
CS2 = ax.contour(CS, colors='k')
ax.set_xlabel(r'$x$ [m]', fontsize=fontSize, labelpad=labelPadX)
ax.set_ylabel(r'$y$ [m]', fontsize=fontSize, labelpad=labelPadY)
plt.yticks(np.arange(0, plateLength+1, 1), fontsize=fontSize)
plt.xticks(np.arange(0, plateWidth+1, 1), fontsize=fontSize)
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('Temperature [C]', fontsize=fontSize,
                    labelpad=labelPadX)
cbar.ax.tick_params(size=0, width=lineWidth)
cbar.add_lines(CS2)
cbar.ax.tick_params(labelsize=fontSize)
ax.tick_params(which='both', direction='in', length=6,
                width=lineWidth, gridOn=False, pad=tickPad)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
plt.setp(list(ax.spines.values()), linewidth=lineWidth)
ax.spines['bottom'].set_color(darkBlue)
ax.spines['top'].set_color(darkBlue)
ax.spines['right'].set_color(darkBlue)
ax.spines['left'].set_color(darkBlue)
plt.show()

