function NumerVelEmCntr_frt = Vel_Darcyflux_frt(u, ny)
NumerVelEmCntr_frt = zeros(ny, 1);
for i = 1:ny
    NumerVelEmCntr_frt(i) = (1/2)*(u(i)+u(i+1));
end