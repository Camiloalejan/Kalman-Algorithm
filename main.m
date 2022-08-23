% Hernández Salazar Camilo Alejandro - 214019774
%% Regresión lineal Adaline FKE
clc; clear; close all; 
load data_lineal
xd = xp; d = yp';
nk = length(xd);
n = 2; w = rand(n,1);
eta = 0.1; P = eye(n); Q = eye(n)*0.01; R = 0.001; 

for i = 1:100
    for k=1:nk
        x = [1; xd(k)];
        y = w'*x;
        e = d(k)-y;

        H = x;
        K = P*H*(R+H'*P*H)^(-1);
        P = P - K*H'*P+Q;
        w = w + eta*K*e;
    end
end

plot_gen = [];
for i=1:nk
    x = [1; xd(i)];
    y = w'*x;
    plot_gen = [plot_gen y];
end

figure
hold on
grid on
plot(xp,yp,'ro','LineWidth',1.5,'MarkerSize',10)
plot(xp,plot_gen,'b','LineWidth',3.5,'MarkerSize',15)
xlabel('xp')
ylabel('yp')
legend('Muestras','Regresión lineal')
disp('Para continuar con la regresión no lineal, presionar cualquier tecla...')
pause()

%% Regresión no lineal MLP FKE
clc; clear; close all;
load data_nolineal
xd = xp'; d = yp';
[~,nK] = size(xd);

n = 10; w = rand(n,1);
eta = 0.25; P = eye(n); Q = eye(n)*0.1; R = 10; 

for i = 1:1000
    for k = 1:nK
        wO = [w(1:2) w(3:4) w(5:6)];
        wS = w(7:10);
        
        xO = [1; xd(k)];
        vO = wO'*xO;
        yO = tanh(vO);
        
        xS = [1; yO];
        vS = wS'*xS;
        yS = vS;
        
        % H
        yS_vS = 1;
        yO_vO = ((ones(3,1)-yO).*(ones(3,1)+yO));
        
        yS_wO1 = yS_vS*wS(1)*yO_vO(1)*xO;
        yS_wO2 = yS_vS*wS(2)*yO_vO(2)*xO;
        yS_wO3 = yS_vS*wS(3)*yO_vO(3)*xO;
        yS_wS = yS_vS*xS;
        
        H = [yS_wO1; yS_wO2; yS_wO3; yS_wS];
        
        % Kalman
        K = P*H*(R+H'*P*H)^(-1);
        p = P - K*H'*P + Q;
        e = d(k)-yS;
        
        w = w + eta*K*e;
    end
end

plot_gen = [];
for k = 1:nK
    wO = [w(1:2) w(3:4) w(5:6)];
    wS = w(7:10);

    xO = [1; xd(k)];
    vO = wO'*xO;
    yO = tanh(vO);

    xS = [1;yO];
    vS = wS'*xS;
    yS = vS;
    
    plot_gen = [plot_gen, yS];
end


figure
hold on
grid on
plot(xp,yp,'ro','LineWidth',1.5,'MarkerSize',10)
plot(xp,plot_gen','b','LineWidth',3.5,'MarkerSize',15)
xlabel('xp')
ylabel('yp')
legend('Muestras','Regresión lineal')
disp('Para continuar con la clasificación, presionar cualquier tecla...')
pause()

%% Clasificación MLP-KFE
clc; clear; close all;
sigmoide = @(v) 1./(1+exp(-v));
load data
xd = x; d = y;
[~,nK] = size(xd);
meanXd = mean(xd')'; maxXd = max(xd')'; minXd = min(xd')';
for i = 1:nK
    xd(:,i) = (xd(:,i)-meanXd)./(maxXd-minXd);
end
n = 13; w = rand(n,1);
eta = 0.2; P = eye(n); Q = eye(n)*0.01; R = 0.001; 

for i = 1:100
    for k = 1:nK
        wO = [w(1:3) w(4:6) w(7:9)];
        wS = w(10:13);
        
        xO = [1; xd(:,k)];
        vO = wO'*xO;
        yO = tanh(vO);
        
        xS = [1; yO];
        vS = wS'*xS;
        yS = sigmoide(vS);
        
        % H
        yS_vS = yS.*(1-yS);
        yO_vO = ((ones(3,1)-yO).*(ones(3,1)+yO));
        
        yS_wO1 = yS_vS*wS(1)*yO_vO(1)*xO;
        yS_wO2 = yS_vS*wS(2)*yO_vO(2)*xO;
        yS_wO3 = yS_vS*wS(3)*yO_vO(3)*xO;
        yS_wS = yS_vS*xS;
        
        H = [yS_wO1; yS_wO2; yS_wO3; yS_wS];
        
        % Kalman
        K = P*H*(R+H'*P*H)^(-1);
        p = P - K*H'*P + Q;
        e = d(k)-yS;
        
        w = w + eta*K*e;
    end
end

for k = 1:nK
    wO = [w(1:3) w(4:6) w(7:9)];
    wS = w(10:13);

    xO = [1; xd(:,k)];
    vO = wO'*xO;
    yO = tanh(vO);

    xS = [1; yO];
    vS = wS'*xS;
    yS = sigmoide(vS);
    
    hold on
    grid on
    if d(k)
        plot(x(1,k),x(2,k),'b*','LineWidth',1,'MarkerSize',10)
    else
        plot(x(1,k),x(2,k),'r*','LineWidth',1,'MarkerSize',10)
    end

    if yS>=0.5
        plot(x(1,k),x(2,k),'bo','LineWidth',1,'MarkerSize',10)
    else
        plot(x(1,k),x(2,k),'ro','LineWidth',1,'MarkerSize',10)
    end
end

xlabel('x_1')
ylabel('x_2')