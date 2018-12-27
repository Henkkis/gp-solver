
clear
close all

N = 200;
L=2;
num_particles =200;
method = 'jacobi';
rerr = 1e-7;

num_splits = 3;

tic
NC = round(N/(2^num_splits));

psi=complex(rand(NC^3,1),rand(NC^3,1));
psi = psi(:);
[psi,u,time1,time2]=gp_solver(NC,L,num_particles,method,rerr,psi);
for id = 1:num_splits
psi = InterPol(NC,round(N/2^(num_splits-id)),psi,L);
NC = round(N/2^(num_splits-id));
[psi,u,time1,time2]=gp_solver(NC,L,num_particles,method,rerr,psi);
end
toc
