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
y = [in.Pos_x in.Pos_y];  

%---------------------------------------------------------------------
% EKF
%---------------------------------------------------------------------
R = cov(y(1:100,:));

%GQG = G*Q*G';
qxy = 2E-4;  qa = 1E-4;  qKr = 5E-4;  qv = 5E-3;
GQG = [qxy 0 0 0 0; 0 qxy 0 0 0; 0 0 qa 0 0; 0 0 0 qKr 0; 0 0 0 0 qv];
   
Cj=[1 0 0 0 0; 0 1 0 0 0];                 % Ausgabematric Kalman-Filter

% Init: 
x_dach = [y(1,1);y(1,2);0;0;0];
P_dach = [.1 0 0 0 0; 0 .1 0 0 0; 0 0 1e-2 0 0; 0 0 0 1e-3 0; 0 0 0 0 1e-2];

for k=1:length(y)
    dy = y(k,:)' - Cj*x_dach;
    M = Cj*P_dach*Cj' + R;
    invM = 1/(M(1)*M(4)-M(2)*M(3))*[M(4) -M(2); -M(3) M(1)]; %invM = pinv(M);
    K = P_dach*Cj'*invM;     %K = P_dach*Cj'*pinv(M);
    x_tilde = x_dach + K*dy;
    P_tilde = (eye(length(x_dach))-K*Cj)*P_dach*(eye(length(x_dach))-K*Cj)' + K*R*K'; 
    
    xP(k)=x_tilde(1); yP(k)=x_tilde(2); alpha(k)=x_tilde(3); Kr(k)=x_tilde(4); v(k)=x_tilde(5); 

    x_dach = [xP(k) - v(k)*Ts*sin(alpha(k));    
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

%---------------------------------------------------------------------
% Ausgabe
%---------------------------------------------------------------------
figure(1); clf; plot(y(:,1),y(:,2),'bo',xP,yP,'g-*',GT.x,GT.y,'r'); axis equal;
figure(2); clf; 
subplot(311); plot(t,alpha,'g-',t,GT.alpha,'r');
subplot(312); plot(t,Kr,'g-',t,GT.Kr,'r');
subplot(313); plot(t,v,'g-',t,GT.v,'r');





%=========================================================================
