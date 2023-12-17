%---------------------------------------------------------------------
% Messdaten
%---------------------------------------------------------------------
clearvars;

in = readtable('xy.dat','Delimiter','space');
GT.x = in.GT_x';
GT.y = in.GT_y';
GT.alpha = in.GT_alpha';
GT.Kr = in.GT_Kr';
GT.v = in.GT_v';
t = in.time; Ts = t(2)-t(1);
aa = in.GT_alpha+sqrt(1e-2)*randn(length(t),1);
y = [in.Pos_x in.Pos_y]';  
%y = [in.Pos_x, in.Pos_y, aa];  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   V O R A B B E S T I M M U N G   V O N   K0   u n d   H
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R0  = 1;
Q0  = 11;

lambda = Ts*sqrt(Q0/R0);
K1 = -1/8*(lambda.^2 + 8*lambda - (lambda+4).*sqrt(lambda.^2+8*lambda));
K2 = .25*(lambda.^2 + 4*lambda - lambda.*sqrt(lambda.^2+8*lambda))/Ts;

K0= [K1;K2]; 
%Ad = [1 Ts; 0 1]; 
%C  = [1 0]
%H = (eye(length(Ad)) - K0*C)*Ad;
H = [1-K1  Ts-K1*Ts; -K2  1-K2*Ts];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   I N I T   R O S E - F I L T E R
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gamma   = 0.9;         % Factor for measurement noise 
Alpha_R = 0.08;        % Smoothing factor measurement noise

%---------------------------------------------------------------------
% EKF
%---------------------------------------------------------------------
R = Gamma*cov(y(:,1:100)'); 

%Q = [5E-4 0; 0 5E-3;];        % Kovarianz Systemrauschen
%G = [0 0; 0 0; 0 0; 1 0; 0 1];
%GQG = G*Q*G';
%qxy = 2E-5;  qa = 4E-4;  qKr = 3E-4;  qv = 5E-3;
qxy = 2E-5;  qa = 4E-4;  qKr = 3E-4;  qv = 5E-3;
GQG = [qxy 0 0 0 0; 0 qxy 0 0 0; 0 0 qa 0 0; 0 0 0 qKr 0; 0 0 0 0 qv];

Cj=[1 0 0 0 0; 0 1 0 0 0];                 % Ausgabematric Kalman-Filter

% Init: 
x_dach = [y(1,1);y(2,1);0;0;0];
P_dach = [.1 0 0 0 0; 0 .1 0 0 0; 0 0 1e-2 0 0; 0 0 0 1e-3 0; 0 0 0 0 1e-2];
%MM = Cj*P_dach*Cj';

x1 = [y(1,1); 0;];   
x2 = [y(2,1); 0;];   


for k=1:length(y)
    x1 = H*x1 + K0*y(1,k);   xR(k)=x1(1);
    x2 = H*x2 + K0*y(2,k);   yR(k)=x2(1);
    R = Gamma*Alpha_R*[y(:,k)-[x1(1);x2(1)]]*[y(:,k)-[x1(1);x2(1)]]' + (1-Alpha_R)*R;
    % R(2)=0; R(3)=0;
    r1(k)=R(1); r2(k)=R(4); 
    
    dy = y(:,k) - Cj*x_dach;
    M = Cj*P_dach*Cj' + R;
    
%    Alpha_M = .001;           % Smoothing factor process noise
%    MM = Alpha_M.*dy*dy' + (1-Alpha_M).*MM;  
%    M1(k)=MM(1); M2(k)=MM(2); M3(k)=MM(3); M4(k)=MM(4);
%    M = MM;
       
    invM = 1/(M(1)*M(4)-M(2)*M(3))*[M(4) -M(2); -M(3) M(1)]; %invM = pinv(M);
    K = P_dach*Cj'*invM;     
%    K = P_dach*Cj'*pinv(M);
    x_tilde = x_dach + K*dy;
    %P_tilde = (eye(length(x_dach)) - K*Cj)*P_dach;
    P_tilde = (eye(length(x_dach))-K*Cj)*P_dach*(eye(length(x_dach))-K*Cj)' + K*R*K'; 
        
%    if x_tilde(5)<0
%        x_tilde(3) = pi+x_tilde(3);
%        x_tilde(5) = -x_tilde(5);
%    end
    
    xP(k)=x_tilde(1); yP(k)=x_tilde(2); alpha(k)=x_tilde(3); Kr(k)=x_tilde(4); v(k)=x_tilde(5); 

    x_dach = [
              xP(k) - v(k)*Ts*sin(alpha(k));    
              yP(k) + v(k)*Ts*cos(alpha(k));    
%              xP(k) - v(k)*Ts*sin(alpha(k)+.5*v(k)*Ts*Kr(k));    
%              yP(k) + v(k)*Ts*cos(alpha(k)+.5*v(k)*Ts*Kr(k));    
              alpha(k) + v(k)*Ts*Kr(k);
              Kr(k);
              v(k)];

    Aj = [1  0  -v(k)*Ts*cos(alpha(k))    0     -Ts*sin(alpha(k));
          0  1  -v(k)*Ts*sin(alpha(k))    0      Ts*cos(alpha(k));
          0  0          1              v(k)*Ts       Kr(k)*Ts;
          0  0          0                 1              0;
          0  0          0                 0              1];
      
%    Aj = [
%          1,  0,  -v(k)*Ts*cos(alpha(k)+0.5*v(k)*Ts*Kr(k)),  -.5*v(k)^2*Ts^2*cos(alpha(k)+0.5*v(k)*Ts*Kr(k)),  -.5*Kr(k)*v(k)*Ts*cos(alpha(k)+0.5*v(k)*Ts*Kr(k))-Ts*sin(alpha(k)+0.5*v(k)*Ts*Kr(k));
%          0,  1,  -v(k)*Ts*sin(alpha(k)+0.5*v(k)*Ts*Kr(k)),  -.5*v(k)^2*Ts^2*sin(alpha(k)+0.5*v(k)*Ts*Kr(k)),  -.5*Kr(k)*v(k)*Ts*sin(alpha(k)+0.5*v(k)*Ts*Kr(k))+Ts*cos(alpha(k)+0.5*v(k)*Ts*Kr(k));
%          0  0          1              v(k)*Ts       Kr(k)*Ts;
%          0  0          0                 1              0;
%          0  0          0                 0              1];

      % Gd = Aj*G;
    % P_dach = Aj*P_tilde*Aj' + Gd*Q*Gd';  
    P_dach = Aj*(P_tilde + GQG)*Aj';  
end

time = t;
Pos_x = in.Pos_x;
Pos_y = in.Pos_y;
GT_x = in.GT_x;
GT_y = in.GT_y;
GT_alpha = in.GT_alpha;
GT_Kr = in.GT_Kr;
GT_v = in.GT_v;
ROSE_x = xP';
ROSE_y = yP';
ROSE_alpha = alpha';
ROSE_Kr = Kr';
ROSE_v = v';


% Init: 
%qxy = 2E-4;  qa = 1E-4;  qKr = 1E-4;  qv = 8E-4;
qxy = 2E-5;  qa = 4E-4;  qKr = 3E-4;  qv = 5E-3;
GQG = [qxy 0 0 0 0; 0 qxy 0 0 0; 0 0 qa 0 0; 0 0 0 qKr 0; 0 0 0 0 qv];
R = cov(y(:,1:100)'); 
x_dach = [y(1,1);y(2,1);0;0;0];
P_dach = [.1 0 0 0 0; 0 .1 0 0 0; 0 0 1e-2 0 0; 0 0 0 1e-3 0; 0 0 0 0 1e-2];

for k=1:length(y)
    dy = y(:,k) - Cj*x_dach;
    M = Cj*P_dach*Cj' + R;
    invM = 1/(M(1)*M(4)-M(2)*M(3))*[M(4) -M(2); -M(3) M(1)]; %invM = pinv(M);
    K = P_dach*Cj'*invM;     
    x_tilde = x_dach + K*dy;
    P_tilde = (eye(length(x_dach))-K*Cj)*P_dach*(eye(length(x_dach))-K*Cj)' + K*R*K'; 
    xP(k)=x_tilde(1); yP(k)=x_tilde(2); alpha(k)=x_tilde(3); Kr(k)=x_tilde(4); v(k)=x_tilde(5); 

    x_dach = [
              xP(k) - v(k)*Ts*sin(alpha(k));    
              yP(k) + v(k)*Ts*cos(alpha(k));    
              alpha(k) + v(k)*Ts*Kr(k);
              Kr(k);
              v(k)];

    Aj = [1  0  -v(k)*Ts*cos(alpha(k))    0     -Ts*sin(alpha(k));
          0  1  -v(k)*Ts*sin(alpha(k))    0      Ts*cos(alpha(k));
          0  0          1              v(k)*Ts       Kr(k)*Ts;
          0  0          0                 1              0;
          0  0          0                 0              1];
      
    P_dach = Aj*(P_tilde + GQG)*Aj';  
end

EKF_x = xP';
EKF_y = yP';
EKF_alpha = alpha';
EKF_Kr = Kr';
EKF_v = v';

figure(1); clf; plot(y(1,:),y(2,:),'bo',xP,yP,'g-*',GT.x,GT.y,'r',xR,yR,'b'); axis equal;
figure(2); clf; 
subplot(311); plot(t,alpha,'g-',t,ROSE_alpha,'b-',t,GT.alpha,'r');
subplot(312); plot(t,Kr,'g-',t,ROSE_Kr,'b-',t,GT.Kr,'r');
subplot(313); plot(t,v,'g-',t,ROSE_v,'b-',t,GT.v,'r');
figure(3); clf; plot(t,r1,t,r2);

rms_KF_xy  = rms(sqrt((xR-GT.x).^2+(yR-GT.y).^2)); %display(rms_KF_xy);
rms_EKF_xy = rms(sqrt((xP-GT.x).^2+(yP-GT.y).^2)); %display(rms_EKF_xy);
rms_ROSE_xy = rms(sqrt((ROSE_x'-GT.x).^2+(ROSE_y'-GT.y).^2)); %display(rms_EKF_xy);
display(['rms_xy KF:', num2str(rms_KF_xy),' EKF:',num2str(rms_EKF_xy),' ROSE:',num2str(rms_ROSE_xy)]);

rms_EKF_a = rms(EKF_alpha'-GT.alpha); %display(rms_EKF_xy);
rms_ROSE_a = rms(ROSE_alpha'-GT.alpha); %display(rms_EKF_xy);
display(['rms_alpha EKF:', num2str(rms_EKF_a),' ROSE:',num2str(rms_ROSE_a)]);

rms_EKF_Kr = rms(EKF_Kr'-GT.Kr); %display(rms_EKF_xy);
rms_ROSE_Kr = rms(ROSE_Kr'-GT.Kr); %display(rms_EKF_xy);
display(['rms_Kr EKF:', num2str(rms_EKF_Kr),' ROSE:',num2str(rms_ROSE_Kr)]);

rms_EKF_v = rms(EKF_v'-GT.v); %display(rms_EKF_xy);
rms_ROSE_v = rms(ROSE_v'-GT.v); %display(rms_EKF_xy);
display(['rms_v EKF:', num2str(rms_EKF_v),' ROSE:',num2str(rms_ROSE_v)]);

display(['Verbesserung EKF/ROSE: xy:', num2str(rms_EKF_xy/rms_ROSE_xy),...
    ' alpha:', num2str(rms_EKF_a/rms_ROSE_a),...
    ' Kr:' num2str(rms_EKF_Kr/rms_ROSE_Kr),...
    ' v:' num2str(rms_EKF_v/rms_ROSE_v),...
    ' Mittel:' num2str(mean([rms_EKF_xy/rms_ROSE_xy,rms_EKF_a/rms_ROSE_a,...
                       rms_EKF_Kr/rms_ROSE_Kr,rms_EKF_v/rms_ROSE_v]))]);


writetable(table(time,Pos_x,Pos_y,GT_x,GT_y,GT_alpha,GT_Kr,GT_v,ROSE_x,...
       ROSE_y,ROSE_alpha,ROSE_Kr,ROSE_v,EKF_x,EKF_y,EKF_alpha,EKF_Kr,EKF_v),...
       'EKF_ROSE.dat','Delimiter','space');
%mean([rms_EKF_xy, rms_EKF_a, rms_EKF_Kr, rms_EKF_v])
%===================, ======================================================
