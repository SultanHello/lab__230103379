import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

public class Lab2Pi {
    static double pi(long n, ForkJoinPool pool) {
        final double dx=1.0/n;
        return pool.submit(() -> LongStream.range(0,n).parallel().mapToDouble(i -> {
            double x=(i+0.5)*dx; return 4.0/(1.0+x*x);
        }).sum()).join()*dx;
    }
    static double serialPi(long n) {
        double dx=1.0/n,sum=0.0;
        for(long i=0;i<n;i++){double x=(i+0.5)*dx;sum+=4.0/(1+x*x);}
        return sum*dx;
    }
    static double naive(long n,int p) throws Exception {
        double[] sum={0}; Thread[] ts=new Thread[p];
        for(int t=0;t<p;t++) { final long a=n*t/p,b=n*(t+1)/p; ts[t]=new Thread(() -> {
            for(long i=a;i<b;i++){ double x=(i+0.5)/n; sum[0]+=4.0/(1+x*x); }
        }); ts[t].start(); }
        for(Thread t:ts)t.join(); return sum[0]/n;
    }
    static double critical(long n,int p) throws Exception {
        double[] sum={0}; Object lock=new Object(); Thread[] ts=new Thread[p];
        for(int t=0;t<p;t++) { final long a=n*t/p,b=n*(t+1)/p; ts[t]=new Thread(() -> {
            for(long i=a;i<b;i++){ double x=(i+0.5)/n, v=4.0/(1+x*x); synchronized(lock){sum[0]+=v;} }
        }); ts[t].start(); }
        for(Thread t:ts)t.join(); return sum[0]/n;
    }
    static long timed(Runnable r){long a=System.nanoTime();r.run();return System.nanoTime()-a;}
    public static void main(String[] args) throws Exception {
        long n=args.length>0?Long.parseLong(args[0]):20_000_000L;
        int trials=args.length>1?Integer.parseInt(args[1]):3;
        System.out.println("kind,p,trial,n,time_ms,pi,abs_error");
        long s=System.nanoTime(); double serial=serialPi(n);
        System.out.printf(Locale.US,"serial,1,1,%d,%.3f,%.15f,%.4e%n",n,(System.nanoTime()-s)/1e6,serial,Math.abs(serial-Math.PI));
        for(int p:new int[]{1,2,4,8,16}) { ForkJoinPool pool=new ForkJoinPool(p); pi(Math.min(n,10000),pool);
            for(int t=1;t<=trials;t++){ final double[] v={0}; long ns=timed(()->v[0]=pi(n,pool));
                System.out.printf(Locale.US,"reduction,%d,%d,%d,%.3f,%.15f,%.4e%n",p,t,n,ns/1e6,v[0],Math.abs(v[0]-Math.PI)); }
            pool.shutdown(); }
        long nr=Math.min(n,1_000_000L);
        for(int p:new int[]{1,2,4,8}) for(int t=1;t<=trials;t++) { final double[] v={0}; long ns=timed(()->{try{v[0]=naive(nr,p);}catch(Exception e){throw new RuntimeException(e);}});
            System.out.printf(Locale.US,"race,%d,%d,%d,%.3f,%.15f,%.4e%n",p,t,nr,ns/1e6,v[0],Math.abs(v[0]-Math.PI)); }
        for(int t=1;t<=trials;t++){final double[] v={0};long ns=timed(()->{try{v[0]=critical(nr,4);}catch(Exception e){throw new RuntimeException(e);}});
            System.out.printf(Locale.US,"critical,4,%d,%d,%.3f,%.15f,%.4e%n",t,nr,ns/1e6,v[0],Math.abs(v[0]-Math.PI));}
    }
}
