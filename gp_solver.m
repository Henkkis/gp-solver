

function [psi,u,time1,time2] = gp_solver(N,L,num_particles,method,err,psi_init)

tic;
%% Input Parameters
%%----------------------------------------------------------------------------------

%% Grid spacing
h = L/(N-1);

%%----------------------------------------------------------------------------------



%% Position vectors
x = linspace(-L/2,L/2,N);
[Yi,Xi,Zi] = meshgrid(linspace(-L/2,L/2,N),linspace(-L/2,L/2,N),linspace(-L/2,L/2,N));

%% Some constants to speed things up
N2 = N*N;
N3 = N*N*N;
%% scaling factor for integration
sc = 1/((N-1)/L);



%% Boundry indices
bottom = (1:N2)';
top = (((N-1)*(N2)+1 ):N3)';
side1 = repmat(1:N,N,1) +repmat(((0:N-1)*N2)',1,N);
side2 = repmat( (N2-N+1):N2   ,N,1)+repmat(((0:N-1 )*N2)',1,N );
side3 = repmat( (1:N:(N2-N+1)),N,1 )+ repmat( ((0:N-1 )*N2)',1,N );
side4 = repmat( (N:N:(N2)),N,1)+repmat( ((0:N-1)*N2)',1,N );
side1=side1(:);
side2=side2(:);
side3=side3(:);
side4=side4(:);

boundry = unique([bottom,top,side1,side2,side3,side4]);
active = setdiff(1:N3,boundry);

%% Laplacian operator

 ys = [active,active,active,active,active,active,active];
 xs = [active,active+1,active-1,active+N,active-N,active+N2,active-N2];
 sval = [-6/h^2*ones(1,length(active)),1/h^2*ones(1,length(active)*6)];


LP = sparse(ys,xs,sval,N3,N3);



%% Strength of the potential
lx = 20 ;
ly = 20 ;
lz = 20 ;


%% Potential function (Harmonic potential)
potfun = @(x,y,z) 1/2*(lx^2.*x.^2 +ly^2.*y.^2+lz^2.*z.^2);
V = potfun(Xi(:,:,:),Yi(:,:,:),Zi(:,:,:));
V = spdiags(V(:),0,N3,N3);

%% Initial wave function
%psi=complex(rand(N,N,N),rand(N,N,N));
%psi = psi(:);
psi = psi_init;


%% Density term
Dens =  abs(psi).^2;

%% Construct the hamiltonian
H0 = -0.5*LP+V;
H = H0 +spdiags(Dens,0,N3,N3) ;
%[psi_s,u_start]=eigs(H);
%psi = psi_s(:,1);
%u = u_start(1,1);

%% Calculate the chemcial potential
%slover using trapez integration, also using FD -> no values between points -> dot product is ok.
%u = trapz(x,trapz(x,trapz(x,reshape(conj(psi).*H*psi,[N,N,N]))))/(trapz(x,trapz(x,trapz(x,reshape(conj(psi).*psi,[N,N,N])))));
u = (psi'*H*psi)/(psi'*psi);
u = real(u);

%% Relaxation parameter for the SOR-method
w = 0.9;

%% Normalize the wavefunction ( and set boundry to zero)
tot = psi'*psi*sc*sc*sc;
psi = psi/sqrt(tot)*sqrt(num_particles);
psi(boundry)=0;

%% max number of iterations
max_iter = 1000;

%% Iteration loop
time1 = toc;
switch method
	case 'sor'
	
		for iter = 1:max_iter
			%% Update the hamiltoninan
			H = H0+spdiags(Dens,0,N3,N3);
			for i = active
				elemH = w/H(i,i);
				psi(i) = psi(i)*(elemH*u+1)-elemH*H(i,:)*psi;
			end
			%% Normalize the wavefunction
			tot = psi'*psi*sc*sc*sc;
			psi = psi/sqrt(tot)*sqrt(num_particles);
			%% Update the density
			Dens =  abs(psi).^2;
			%% Update the chemical potential
			
			u = (psi'*H*psi)/(psi'*psi);
			u = real(u);
			
			if(sum( abs(H*psi-u*psi))/sum(abs(H*psi)) < err  )
				break;	
			end
			

		end

	case 'jacobi'
		s_num= sqrt(num_particles);
		sc = h^3;
      H0d = diag(H0);
        psi= gpuArray(psi);
        H0 = gpuArray(H0);
        Dens = gpuArray(Dens);
        u = gpuArray(u);
        H0d = gpuArray(H0d);
        
		for iter =1:max_iter
			psi_old = psi;
			%% Update the hamiltoninan
			
				

			
			pd = psi.*Dens;
			
			hPsi = H0*psi+pd;
			pp = psi'*psi;
			u = real(psi'*(hPsi)/(pp));
            pu = psi*u;
            cer = (hPsi-pu)'*(hPsi-pu) /(pu'*pu); 
			
            if(  cer    < err  )
				break;	
			end


			iH = 1./(H0d+Dens);
			psi=psi_old.*(u*iH+1)- iH.*hPsi;

			%% Normalize the wavefunction
			%% Update the density
			Dens =  abs(psi).^2;
			%% Update the chemical potential			
			
		end

	otherwise
		warning('Iteration method not recoginsed.')
end
time2=toc;
tot = real(psi'*psi*sc);
psi = psi/sqrt(tot)*s_num;
psi = gather(psi);
end
