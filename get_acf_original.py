import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import curve_fit 
from scipy.integrate import cumtrapz, trapz
from scipy.signal import find_peaks , argrelextrema
from more_itertools import chunked
import time
import sys
import cmath

print(time.asctime(time.localtime(time.time())))
b_time = time.time()

# get autocorrelation <x,x>
def get_ac(x):
    Qf = fft(x)
    k = len(x)
    Qac = np.real(ifft(Qf * np.conj(Qf)) / k)
    return Qac
#get cross-correlation <x,y>
def get_xac(x,y):
    if len(x) != len(y):
        raise Exception('x and y should have the same length')
    Qnf = fft(x)
    Qtf = fft(y) 
    k = len(x)
    Qac = np.real(ifft(Qtf*np.conj(Qnf))/k)
    return Qac
    
def read_atoms(natoms, initial_file):
    init = np.loadtxt(initial_file, skiprows = 24 , max_rows = natoms )
    atom_type = init[:, 2].astype(int)
    return atom_type
    
def read_pos(eignatoms, eigfile):
    positions = np.loadtxt( eigfile , skiprows = 1 , max_rows=eignatoms)
    atom_pos = positions[:,1:]
    return atom_pos
    
def read_dump_file( N , natoms, eignatoms , dp ):
    # ux = np.zeros((N,eignatoms))
    # uy = np.zeros((N,eignatoms))
    # uz = np.zeros((N,eignatoms))
    vx = np.zeros((N,eignatoms))
    vy = np.zeros((N,eignatoms))
    vz = np.zeros((N,eignatoms))
    for n in range(N):
        data = np.array( dp[ n*natoms: n*natoms+natoms , :] )
        for i in range(min(natoms,eignatoms)):
            # ux[ n , i ] = data[ i , 1 ]
            # uy[ n , i ] = data[ i , 2 ]
            # uz[ n , i ] = data[ i , 3 ]
            vx[ n , i ] = data[ i , 0 ]
            vy[ n , i ] = data[ i , 1 ]
            vz[ n , i ] = data[ i , 2 ]
    return vx, vy ,vz

def read_eig_file( M , eignatoms , eigfile ):
    eigx = np.zeros((M,eignatoms),dtype = np.complex64)
    eigy = np.zeros((M,eignatoms),dtype = np.complex64)
    eigz = np.zeros((M,eignatoms),dtype = np.complex64)
    for m in range(M):
        eigx[m,:] = eig[ m*(eignatoms) :  m*(eignatoms)+eignatoms , 0]+ 1j*eig[ m*(eignatoms) :  m*(eignatoms)+eignatoms , 3]
        eigy[m,:] = eig[ m*(eignatoms) :  m*(eignatoms)+eignatoms , 1]+ 1j*eig[ m*(eignatoms) :  m*(eignatoms)+eignatoms , 4]
        eigz[m,:] = eig[ m*(eignatoms) :  m*(eignatoms)+eignatoms , 2]+ 1j*eig[ m*(eignatoms) :  m*(eignatoms)+eignatoms , 5]
    return eigx , eigy , eigz

def calc_Am( N, M, natoms, eignatoms, dp , eig  ):
    vx, vy, vz = read_dump_file( N , natoms, eignatoms , dp )  
    eigx , eigy , eigz = read_eig_file( M , eignatoms , eig)
    # Am =  np.abs(np.dot(ux, np.transpose(eigx)))**2 + np.abs(np.dot(uy, np.transpose(eigy)))**2 + np.abs(np.dot(uz, np.transpose(eigz)))**2   #A^2/ps
    Amdot = np.abs(np.dot(vx, np.transpose(eigx)))**2 + np.abs(np.dot(vy, np.transpose(eigy)))**2 + np.abs(np.dot(vz, np.transpose(eigz)))**2
    print('Shape:', np.shape(Amdot)) 
    return Amdot
    
def calc_Etm( N, M, Am , Amdot,freq):
    Etm = np.zeros((N-1,M))
    for m in range(M):
        Etm[:,m] = 2*0.029979*np.pi*freq[m]*Am[1:N,m]/2 + Amdot[1:N,m]/2
    return Etm 

def calc_acf( Etm ):
    num = int( len(Etm) / 2 ) 
    acf_Etm = np.zeros( ( num , M ) )
    for m in range(M):
        acf_Etm[ : , m ] = get_ac( Etm[ : , m ] )[0:num]
        acf_Etm[ : , m ] = acf_Etm[ : , m ] / acf_Etm[0,m]
    return acf_Etm 

def func( x ,tau ,a ,b):
    return a*np.exp(-x/tau)+b

def funcum( x ,tau ,a ):
    return a*(1-np.exp(-x/tau))

def get_ptp_low(ave_acf,chunk_time, dt, ave_time,m):
    chunks = np.arange( 0 , len(ave_acf)+1, int(chunk_time/dt) )
    # print( "len(chunks) = " , len(chunks) )
    time = np.arange( 0, len(ave_acf)*dt, chunk_time )
    t = np.arange(0, len(ave_acf)*dt , dt)
    ptps = np.zeros( len(chunks)-1 )
    for i in range(len(chunks)-1):
        ptps[i] = np.ptp( ave_acf[chunks[i] : chunks[i+1]])
    d_ptps = np.absolute( np.diff(ptps) )
    a = np.array_split(ptps, int( len(ptps)*chunk_time/ave_time))
    d_ave = np.zeros(len (a) )
    for i in range(len(a)):
        d_ave[i] = np.ptp(a[i])
    sort_d = np.sort(d_ave)
    #print( d_ave )
    tail = np.mean(sort_d[:4])
    #print(tail)
    cutoff = np.where(d_ave < 1.08*tail)[0][0]
    return (cutoff+2)*ave_time

def expfit(ave_acf, dt , cutoff ,T,m ):
    t = np.arange(0,len(ave_acf)*dt,dt)
    chunks = np.arange( 0 , len(ave_acf)+1, int(T/dt) )
    time = np.arange( T/2, len(ave_acf)*dt-T/2, T )
    ptps = np.zeros( len(time) )
    for i in range( len(time) ):
        ptps[i] = np.mean( ave_acf[chunks[i] : chunks[i+1]]) #np.ptp( ave_acf[chunks[i] : chunks[i+1]]) 
    parabound = ([0,0,0], [3*cutoff, 1,1])
    #para,_ = curve_fit(func,t[0:int(cutoff/dt)],ave_acf[0:int(cutoff/dt)], bounds = parabound , maxfev=10000 )
    para,_ = curve_fit(func,time[0:int(cutoff/T)],ptps[0:int(cutoff/T)], bounds = parabound , maxfev=10000 )
    tau=para[0]
    # plt.plot( t[:cutoff] , ave_acf[:cutoff] ,alpha = 0.3)
    # plt.plot( t[1:cutoff], func(t[1:cutoff],para[0],para[1],para[2]) , 'k--')
    # plt.title( 'tau = %s' % tau)
    # plt.savefig( 'exps_%s.png' % m )
    # plt.close( )
    return tau

if __name__ == "__main__":
    dt=0.001*8
    #maxtime = 80 #ps
    nm=np.zeros(4)
    eigfile = '../../gulp_eig/result.eig'
    nm[0] = np.loadtxt( 'ThO2.trajectory', skiprows = 3 , max_rows = 1)
    nm[1] = int(sys.argv[1])/(int(nm[0])+9)  
    nm[2] = np.loadtxt( eigfile , max_rows = 1)
    nm[3] = 3 * nm[2]
    nm = nm.astype(int)
    natoms=nm[0]
    N=nm[1]
    eignatoms=nm[2]
    M=nm[3]
    print('natoms=', natoms, 'dt = ', dt ,'dumptime/dt =' ,N , 'eignatoms=' , eignatoms , 'Nmodes=' ,M )

    skip1 = np.arange(9*N)
    for i in range(N):
        skip1[i*9:(i+1)*9]=range(i*(9+natoms),i*(9+natoms)+9)
    Av = pd.read_csv( 'ThO2.trajectory' , skiprows = skip1 , sep = ' ' , header = None , low_memory = False, dtype=np.float64 )
    dp = Av.to_numpy( )  #dump displacement and velocity
    print(np.shape(dp))
    print('Read trajectory Time Used:', time.time()-b_time, 's' )
    
    for i in range(3):
        print('i=',i)
        M = 3*eignatoms
        skip2=np.arange( eignatoms+4+((eignatoms+2)*3*eignatoms+1)*i+2*M )
        for j in range(M):
            skip2[ (eignatoms+4+((eignatoms+2)*3*eignatoms+1)*i+j*2) ] = eignatoms+4+((eignatoms+2)*3*eignatoms+1)*i+j*(eignatoms+2)
            skip2[ (eignatoms+5+((eignatoms+2)*3*eignatoms+1)*i+j*2) ] = eignatoms+5+((eignatoms+2)*3*eignatoms+1)*i+j*(eignatoms+2)
        eig = pd.read_csv( eigfile , skiprows = skip2 , nrows = M*eignatoms , sep='\s+' , header = None , low_memory = False ,  dtype=np.float64)
        eig = eig.to_numpy( ) 
        freq = np.loadtxt('../../gulp_eig/freq.txt', skiprows = M*i, max_rows = M)*0.029979
        Amdot = calc_Am( N, M, natoms, eignatoms, dp , eig  )
        #Etm = calc_Etm(N, M, Am , Amdot, freq)
        acf = calc_acf(Amdot)
        tau = np.zeros(M)
        for m in range(M):
            chunk_time = 0.4 #ps
            ave_time = 2 #ps
            cutoff = get_ptp_low(acf[:,m],chunk_time, dt, ave_time , m)
            #cutoff = 30
            T= int( 1/2/freq[m]/dt)*dt
            # print(m,cutoff)
            tau[m] = expfit( acf[:,m], dt, int(cutoff/dt) ,T,m)
            #if m>324 and tau[m] > 5:
                #print("large", m)
        taus = np.vstack((freq,tau)).T
        np.savetxt( 'taus_ptps_%s.txt' % i, taus, fmt = '%1.6f', header = '# freqs /THz    taus /ps')
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter( taus[:,0], 1/taus[:,1])
        plt.savefig('taus_ptps_%s.png' % i )
        plt.close()
        
        print('Time Used:', time.time()-b_time, 's' )
print('Total Time Used:', time.time()-b_time, 's' )
