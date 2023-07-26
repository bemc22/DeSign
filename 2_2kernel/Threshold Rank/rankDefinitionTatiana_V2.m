function [rango]  = rankDefinitionTatiana_V2(binary, threshold)

mul = binary.*threshold;

%Se obtiene el maximo nivel de threshold que se superó
maximo = max(mul(:));

my_th = [0 ; sort(unique(threshold),'ascend')];

%aux         = abs((mul<0).*mul);
%aux         = aux(aux>0);
%if(~isempty(aux))
%    Ref_min     = min(aux(:));
%    pos         = find(my_th == Ref_min(1));
%    minimo = my_th(pos-1);
%else
    minimo = min(mul(:));
%end

rango = [minimo maximo];

end