function [rango]  = rankDefinitionNoThr_V2(binary)

sigma = 3;

suma = sum(binary(:));

switch suma
    case 4
        rango = [0 sigma];
        
    case -4
        rango = [-sigma 0];
        
    otherwise
        rango = [-sigma sigma];
        

end