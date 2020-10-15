function [tmin, L] = nbldi(Lshp, A, B, C, S0)

setlmis([]);    
                                         
Lv=lmivar(1,Lshp);
%Lv=lmivar(1,[1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1]);

lmiterm([-1 1 1 Lv],1,1);                        % LMI #1: L
lmiterm([2 1 1 Lv],C',C);                        % LMI #2: C'*L*C
lmiterm([2 1 1 0],A'*S0+S0*A);                   % LMI #2: A'*P+P*A
lmiterm([2 2 1 0],B'*S0);                        % LMI #2: B'*P
lmiterm([2 2 2 Lv],1,-1);                        % LMI #2: -L

lmisyst=getlmis;

[tmin,xfeas] = feasp(lmisyst);
L = dec2mat(lmisyst,xfeas,Lv);
