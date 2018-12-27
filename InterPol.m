function [Psi_inter] = InterPol(N_old,N_new,psi,L)

	o_range = linspace(-L/2,L/2,N_old);
	n_range = linspace(-L/2,L/2,N_new);

	[X,Y,Z]=meshgrid(o_range,o_range,o_range);
	[Xq,Yq,Zq] =meshgrid(n_range,n_range,n_range);

	psi = reshape(psi,[N_old,N_old,N_old]);
	Vr = interp3(X,Y,Z,real(psi),Xq,Yq,Zq);
	Vi = interp3(X,Y,Z,imag(psi),Xq,Yq,Zq);
	Psi_inter = Vr+i*Vi;

	Psi_inter = Psi_inter(:);
end
