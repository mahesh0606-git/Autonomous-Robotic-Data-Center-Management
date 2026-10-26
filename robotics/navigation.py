"""Offline 2D occupancy grid and inflated-obstacle A*. NOT a safe robot controller."""
import heapq
import math
import numpy as np

def inflated_grid(occupied,radius_cells):
    grid=np.asarray(occupied,dtype=bool)
    if grid.ndim!=2 or radius_cells<0: raise ValueError('Invalid grid/radius')
    out=grid.copy()
    for y,x in np.argwhere(grid):
        for dy in range(-radius_cells,radius_cells+1):
            for dx in range(-radius_cells,radius_cells+1):
                if dx*dx+dy*dy<=radius_cells*radius_cells:
                    yy,xx=y+dy,x+dx
                    if 0<=yy<grid.shape[0] and 0<=xx<grid.shape[1]:out[yy,xx]=True
    return out

def astar(occupied,start,goal):
    """Return grid cells or None; no dynamic obstacles, kinematics or timing."""
    grid=np.asarray(occupied,dtype=bool); h,w=grid.shape
    def valid(p):return 0<=p[0]<h and 0<=p[1]<w and not grid[p]
    if not valid(start) or not valid(goal):return None
    queue=[(0,start)];dist={start:0};prev={}
    while queue:
        _,u=heapq.heappop(queue)
        if u==goal:
            path=[u]
            while u in prev:u=prev[u];path.append(u)
            return path[::-1]
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
            v=(u[0]+dy,u[1]+dx)
            if valid(v) and dist[u]+1<dist.get(v,math.inf):
                dist[v]=dist[u]+1;prev[v]=u
                heapq.heappush(queue,(dist[v]+abs(v[0]-goal[0])+abs(v[1]-goal[1]),v))
    return None
