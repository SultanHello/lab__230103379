import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class Lab3Mandelbrot {
    static int W=640,H=360,MAX=500;
    static int pixel(int px,int py){double x0=(px-W/2.0)*4.0/W,y0=(py-H/2.0)*4.0/H,x=0,y=0;int it=0;
        while(x*x+y*y<=4&&it<MAX){double q=x*x-y*y+x0;y=2*x*y+y0;x=q;it++;}return it;}
    static class Run {long ns;long[] work;long checksum;}
    static Run run(String policy,int p,int chunk){int[][] img=new int[H][W];AtomicInteger next=new AtomicInteger();long[] work=new long[p];Thread[] ts=new Thread[p];
        long start=System.nanoTime();final Object guidedLock=new Object();
        for(int id=0;id<p;id++){final int tid=id;ts[id]=new Thread(()->{
            if(policy.equals("static")){
                for(int a=tid*chunk;a<H;a+=p*chunk) for(int y=a;y<Math.min(H,a+chunk);y++) for(int x=0;x<W;x++){int v=pixel(x,y);img[y][x]=v;work[tid]+=v;}
            } else {
                while(true){int a,c;
                    if(policy.equals("dynamic")){a=next.getAndAdd(chunk);c=chunk;}
                    else {synchronized(guidedLock){a=next.get();if(a>=H)break;c=Math.max(chunk,(H-a+p*4-1)/(p*4));next.set(a+c);}}
                    if(a>=H)break;
                    for(int y=a;y<Math.min(H,a+c);y++)for(int x=0;x<W;x++){int v=pixel(x,y);img[y][x]=v;work[tid]+=v;}
                }
            }
        });ts[id].start();}
        try{for(Thread t:ts)t.join();}catch(InterruptedException e){throw new RuntimeException(e);}
        Run r=new Run();r.ns=System.nanoTime()-start;r.work=work;long sum=0;for(int[] row:img)for(int v:row)sum+=v;r.checksum=sum;return r;
    }
    public static void main(String[] a){int trials=a.length>0?Integer.parseInt(a[0]):3;System.out.println("policy,p,chunk,trial,width,height,max_iter,time_ms,imbalance,checksum");
        for(String policy:new String[]{"static","dynamic","guided"})for(int p:new int[]{2,4,8,16})for(int c:new int[]{1,16,64,256})for(int t=1;t<=trials;t++){
            Run r=run(policy,p,c);long min=Arrays.stream(r.work).min().orElse(0),max=Arrays.stream(r.work).max().orElse(0),tot=Arrays.stream(r.work).sum();double imb=tot==0?0:(max-min)/(tot/(double)p);
            System.out.printf(Locale.US,"%s,%d,%d,%d,%d,%d,%d,%.3f,%.5f,%d%n",policy,p,c,t,W,H,MAX,r.ns/1e6,imb,r.checksum);
        }
    }
}
