%---------------------------------------------------------------------
% Messdaten
%---------------------------------------------------------------------
clearvars;

D = 0.1;
Ts = .1;
vv = [0*ones(1,100) .2:.2:2 2*ones(1,350) 2.1:.1:3 3*ones(1,100) 2.8:-.2:1 1*ones(1,350) .9:-.1:0 0*ones(1,60)];
kk = [0*ones(1,150) .05:.05:.4 .4*ones(1,30) .35:-.05:0 0*ones(1,80)...
      -.1:-.1:-.8 -.8*ones(1,11) -.7:.1:0 0*ones(1,100)...
      -.02:-.02:-.2 -.2*ones(1,55) -.18:.2:0 0*ones(1,35)...
      .02:.02:.2 .2*ones(1,55) .18:-.2:0 0*ones(1,145)...
      .01:.01:.1 .1*ones(1,165) .09:-.01:0 0*ones(1,100)...
      ];
vl = vv./(1-kk*.5*D);
vr = vv./(1+kk*.5*D);
t  = Ts*(1:length(vv));

xx=0; yy=0; aa=0.3; sl=0.0211; sr=0;
for k=2:length(vv)
    xx(k) = xx(k-1) - vv(k-1)*Ts*sin(aa(k-1));    
    yy(k) = yy(k-1) + vv(k-1)*Ts*cos(aa(k-1));    
    aa(k) = aa(k-1) + vv(k-1)*Ts*kk(k-1);
    sl(k) = sl(k-1) + vl(k-1)*Ts;
    sr(k) = sr(k-1) + vr(k-1)*Ts;
end;

sl = round(100*sl)/100;
sr = round(100*sr)/100;

s = .5*(sl+sr);
aaa = (sl-sr)/D;

%Rxy = [5e-3*ones(2,300) 5e-2*ones(2,400) 1e-1*ones(2,300)]; 
Rxy = 1e-2*[0.5*ones(2,200) 5*ones(2,50) 1*ones(2,100) 5*ones(2,30) 20*ones(2,300) 10*ones(2,220) 1*ones(2,100)]; 

y = [xx; yy]' + [sqrt(Rxy).*randn(2,length(xx));]';

time = t';
Pos_x = y(:,1);
Pos_y = y(:,2);
GT_x = xx';
GT_y = yy';
GT_alpha = aa';
GT_Kr = kk';
GT_v = vv';
writetable(table(time,Pos_x,Pos_y,GT_x,GT_y,GT_alpha,GT_Kr,GT_v),'xy.dat','Delimiter','space');

figure(1); clf; plot(y(:,1),y(:,2),'ro',xx,yy,'gx'); axis equal;
figure(2); clf; 
subplot(311); plot(t,aa,'g');
subplot(312); plot(t,kk,'g');
subplot(313); plot(t,vv,'g');


%[xP; yP; alpha; Kr; v]'



%=========================================================================
