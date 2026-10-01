import os, time, platform, json
import numpy as np
import numba
from numba import njit, prange

RESULTS = {'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'numba_max_threads': numba.config.NUMBA_NUM_THREADS}
print('CPU', RESULTS)

@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return (4.0 * inside_circle) / n_samples

_ = monte_carlo_pi(10_000)
N=120_000_000
counts=sorted(set(t for t in [1,2,4,8,numba.config.NUMBA_NUM_THREADS] if t <= numba.config.NUMBA_NUM_THREADS))
RESULTS['monte_carlo']={}
t1=None
for t in counts:
    numba.set_num_threads(t)
    start=time.perf_counter(); estimate=monte_carlo_pi(N); elapsed=time.perf_counter()-start
    if t==1:t1=elapsed
    RESULTS['monte_carlo'][str(t)]={'seconds':elapsed,'pi_estimate':estimate,'speedup':t1/elapsed,'efficiency_percent':(t1/elapsed/t)*100}
    print('MC',t,RESULTS['monte_carlo'][str(t)],flush=True)

@njit(parallel=True)
def render_rows(h,w,max_iter):
    img=np.zeros((h,w),dtype=np.int32)
    for r in prange(h):
        cy=-1.2+(r/h)*2.4
        for c in range(w):
            cx=-2.0+(c/w)*2.5; zr=0.; zi=0.; it=0
            while zr*zr+zi*zi<=4.0 and it<max_iter:
                nr=zr*zr-zi*zi+cx; zi=2*zr*zi+cy; zr=nr; it+=1
            img[r,c]=it
    return img
@njit(parallel=True)
def render_cols(h,w,max_iter):
    img=np.zeros((h,w),dtype=np.int32)
    for c in prange(w):
        cx=-2.0+(c/w)*2.5
        for r in range(h):
            cy=-1.2+(r/h)*2.4; zr=0.; zi=0.; it=0
            while zr*zr+zi*zi<=4.0 and it<max_iter:
                nr=zr*zr-zi*zi+cx; zi=2*zr*zi+cy; zr=nr; it+=1
            img[r,c]=it
    return img
numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
_=render_rows(100,100,50); _=render_cols(100,100,50)
H=W=2500; MI=1000
start=time.perf_counter(); grid_rows=render_rows(H,W,MI); tr=time.perf_counter()-start
start=time.perf_counter(); grid_cols=render_cols(H,W,MI); tc=time.perf_counter()-start
RESULTS['mandelbrot']={'rows_seconds':tr,'cols_seconds':tc,'faster':'rows' if tr<tc else 'cols','shape':[H,W],'max_iter':MI}
print('MANDELBROT',RESULTS['mandelbrot'],flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8)); plt.imshow(grid_rows,cmap='magma',extent=[-2.,.5,-1.2,1.2]); plt.title(f'Mandelbrot {H}x{W} (Render: {tr:.2f}s)'); plt.axis('off'); plt.savefig('outputs/mandelbrot_output.png',dpi=300,bbox_inches='tight'); plt.close()

@njit(parallel=True)
def heat_step(u,u_next,alpha=0.20):
    rows,cols=u.shape
    for i in prange(1,rows-1):
        for j in range(1,cols-1):
            u_next[i,j]=u[i,j]+alpha*(u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]-4.0*u[i,j])

def heat(dtype):
    n=1500; steps=300
    u=np.zeros((n,n),dtype=dtype); v=np.zeros_like(u)
    u[0,:]=100.; u[:,0]=100.; v[0,:]=100.; v[:,0]=100.
    heat_step(u,v)
    st=time.perf_counter()
    for _ in range(steps):
        heat_step(u,v); u,v=v,u
    elapsed=time.perf_counter()-st
    return elapsed,(n*n*steps)/elapsed/1e6
numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
d64=heat(np.float64); d32=heat(np.float32)
RESULTS['heat']={'float64_seconds':d64[0],'float64_megacells_per_sec':d64[1],'float32_seconds':d32[0],'float32_megacells_per_sec':d32[1],'runtime_factor_float64_over_float32':d64[0]/d32[0],'shape':[1500,1500],'steps':300}
print('HEAT',RESULTS['heat'],flush=True)
with open('outputs/results.json','w') as f: json.dump(RESULTS,f,indent=2)
