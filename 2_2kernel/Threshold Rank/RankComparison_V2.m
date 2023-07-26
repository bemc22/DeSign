clear
close all

pruebas = 100000;
pd = makedist('Normal', 'mu', 0, 'sigma', 1);

Errores = zeros(2,8);

threshold1 = [1 1;3 3];
threshold2 = [1 1;0 3];
threshold3 = [0 0;0 0];
threshold4 = [1 1;1 1];
threshold5 = [3 3;3 3];
threshold6 = [5 5;5 5];
threshold7 = [7 7;7 7];
threshold8 = [9 9;9 9];

thresholds = [threshold1 threshold2 threshold3 threshold4 threshold5 threshold6 threshold7 threshold8];

for h=0:7
    h
    threshold = thresholds(:,2*h+1:2*h+2);
    threshold = halfwave(threshold);
    
    Error = 0;
    ErrorNoThr = 0;
    
    for i=1:pruebas
        %GeneraciÃ³n de matriz normalizada después de convolución
        normalized = pd.random(2,2);
        
        %Llevar a cero todos los valores negativos
        %image(image<0) = 0;
        rangoReal = [min(normalized(:)) max(normalized(:))];
        
        %AplicaciÃ³n de threshold
        app = normalized-threshold;
        binaryapp = binarize(app);
        
        rangoNoThr = rankDefinitionNoThr_V2(binaryapp);
        rangoEstimado = rankDefinitionTatiana_V2(binaryapp, threshold);
        
        Error = Error + sum(abs(rangoReal - rangoEstimado).^2);
        ErrorNoThr  = ErrorNoThr + sum(abs(rangoReal - rangoNoThr).^2);
        
    end
    
    Error = Error/pruebas
    ErrorNoThr = ErrorNoThr/pruebas
    
    
    Errores(:,h+1) = [Error; ErrorNoThr];
end

