#include <cmath>
#include <algorithm>
#include <omp.h>

// No model approximation: the live filter is precomputed and never depends on
// chart parameters. b is live prediction minus protected prediction.
extern "C" void chart(const double* innovation, const double* covariance,
                      const double* response, const double* direction,
                      int n, int horizon, int p, int method, double k1,
                      double g, double k2, int L, double h, int tau,
                      double qlevel, double qslope, int threads,
                      int* rl, int* candidate_at_change) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; ++i) {
        double s[32]={0}, r[32]={0}, bl[32]={0}, bs[32]={0};
        double ap00=0, ap01=0, ap11=0;
        int age=0;
        bool confirming=false;
        rl[i]=horizon+1;
        candidate_at_change[i]=-1;
        for (int t=0; t<horizon; ++t) {
            if (t==tau-1) candidate_at_change[i]=int(confirming);
            const double* c=covariance+t*6;
            const double* e=innovation+(static_cast<long long>(i)*horizon+t)*p;
            double v[32], norm=0;
            if (method==0 || !confirming) {
                for (int j=0; j<p; ++j) {
                    v[j]=s[j]+(e[j]+response[t]*direction[j])/std::sqrt(c[5]);
                    norm+=v[j]*v[j];
                }
                norm=std::sqrt(norm);
                double factor=std::max(0.0,1.0-k1/std::max(norm,1e-12));
                for(int j=0;j<p;++j) s[j]=v[j]*factor;
                double snorm=norm*factor;
                if(method==0) {
                    if(snorm>h){rl[i]=t+1;break;}
                    continue;
                }
                if(snorm>g){
                    confirming=true; age=0;
                    ap00=c[0]; ap01=c[1]; ap11=c[2];
                    for(int j=0;j<p;++j){r[j]=0;bl[j]=0;bs[j]=0;}
                }
            }
            if(!confirming) continue;
            if(age>0){
                ap00=ap00+2*ap01+ap11+qlevel;
                ap01+=ap11; ap11+=qslope;
            }
            norm=0;
            for(int j=0;j<p;++j){
                v[j]=r[j]+(e[j]+response[t]*direction[j]+bl[j])/std::sqrt(ap00+1);
                norm+=v[j]*v[j];
            }
            norm=std::sqrt(norm);
            double factor=std::max(0.0,1.0-k2/std::max(norm,1e-12));
            for(int j=0;j<p;++j) r[j]=v[j]*factor;
            double rnorm=norm*factor;
            ++age;
            // A crossing on the last allowed observation is an alarm, not a reset.
            if(rnorm>h){rl[i]=t+1;break;}
            if((rnorm<=1e-12 && age>=2) || age>=L){
                confirming=false; age=0;
                for(int j=0;j<p;++j){s[j]=0;r[j]=0;}
            } else {
                for(int j=0;j<p;++j){
                    double ej=e[j]+response[t]*direction[j];
                    bl[j]+=bs[j]+(c[3]+c[4])*ej;
                    bs[j]+=c[4]*ej;
                }
            }
        }
    }
}
