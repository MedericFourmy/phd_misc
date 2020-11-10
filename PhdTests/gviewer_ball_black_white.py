import example_robot_data as robex                                      
r=robex.loadSolo(False)                                                 
r.initViewer(loadModel=True)
gv=r.viewer.gui
# gv.deleteNode('world/pinocchio',True)

gv.createGroup('world/angmom')
for i in range(1,5):
    gv.deleteNode('world/angmom/w%d'%i,True)
    gv.deleteNode('world/angmom/b%d'%i,True)
    gv.addSphere('world/angmom/w%d'%i,.02,[1,1,1,1]   )   
    gv.addSphere('world/angmom/b%d'%i,.02,[0,0,0,1])   
eps = 1e-3
gv.applyConfiguration('world/angmom/w1',[ eps, eps, eps ,0,0,0,1])
gv.applyConfiguration('world/angmom/w2',[-eps,-eps, eps ,0,0,0,1])
gv.applyConfiguration('world/angmom/b1',[-eps, eps, eps ,0,0,0,1])
gv.applyConfiguration('world/angmom/b2',[ eps,-eps, eps ,0,0,0,1])
gv.applyConfiguration('world/angmom/b3',[ eps, eps,-eps ,0,0,0,1])
gv.applyConfiguration('world/angmom/b4',[-eps,-eps,-eps ,0,0,0,1])
gv.applyConfiguration('world/angmom/w3',[-eps, eps,-eps ,0,0,0,1])
gv.applyConfiguration('world/angmom/w4',[ eps,-eps,-eps ,0,0,0,1])

gv.applyConfiguration('world/angmom', [0,0,1, 0,0,0,1])

gv.refresh()