function [rango]  = rankDefinition2_V2(binary, threshold)

%Multipliación de threshold para ver cuales valores se superaron y cuales
%no
mul = binary.*threshold;

%Se obtiene el maximo nivel de threshold que se superó
maximo = max(mul(:));

%Se vuelven positivos aquellos thresholds que no se superaron
mulaux = mul.*-1;

%Se ponen infinito los thresholds negativos
mulaux(mulaux<0) = max(threshold(:));

%Se obtiene el maximo de los que no se superaron
rango2 = max(mulaux(:));

%Si el máximo es mayor a cero, es decir, la codificación no fue de todos -1
%entonces se asigna el primer rango a este
if maximo > 0
    rango1 = maximo;
else
    %Si fueron todos -1, entonces se asigna el rango minimo entre 0 y el
    %threshold minimo.
    rango1 = 0;
    rango2 = -maximo;
end

rango = [rango1 rango2];

end