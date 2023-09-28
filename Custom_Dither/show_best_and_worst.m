function [] = show_best_and_worst(metrics, thresholds)

metric_names = ["correlation", "mse", "psnr", "total variation", "wpsnr"];

%cmap = [0 0 0;0 1 1;0 1 1;0 1 0;0 1 0;0 0 1;0 0 1;1 1 0; 1 1 0; 1 1 1];
map = {'#B3D8E5','#85C2D6','#51BDE1','#23A6D1','#2F94C6','#5277B7','#0078E0','#545CF2','#1F1FFF','#000052'};
cmap = validatecolor(map, 'multiple');
% threshold_continuous = 0;

%Para cada métrica
for metric = 1:5
    
    %Se ordena de menor a mayor
    metric_indv = metrics(metric,:);
    [metric_indv_sorted, I] = sort(metric_indv);
    figure(metric)
    tcl = tiledlayout(2,10);
    
    %Se muestran los 5 menores
    for i=1:10
        nexttile()
        imagesc(thresholds(:, :, I(i)))
        colormap(cmap)
        caxis([0 9]) 
        title(num2str(metric_indv_sorted(i)))
        axis off
        
    end
    
    %Se muestran los 5 mayores
    for i=1:10
        nexttile()
        imagesc(thresholds(:, :, I(end-(i-1))))
        colormap(cmap)
        caxis([0 9]) 
        title(num2str(metric_indv_sorted(end-(i-1))))
        axis off
        
    end
    
    % Personalización de gráfica
    cb = colorbar;
    cb.Layout.Tile = 'east';
    % cbscalein = cb.Limits;
    % cbscaleout = [0 5];
    % ticks = linspace(cbscaleout(1),cbscaleout(2),size(cmap,1)+1);
    % cb.Ticks = diff(cbscalein)*(ticks-cbscaleout(1))/diff(cbscaleout) + cbscalein(1);
    cb.Ticks = [0 1 3 5 7 9];
    cb.TickLabels = [0:1:1 3:2:9];
    
    sgtitle(strcat(metric_names(metric), ' top(min)', ' bottom(max)'))
    
    % Ubicación de nuestro threshold
    ours = [1 1; 3 3];
    idx = find_threshold(ours, thresholds);
    oursidx = find(I==38);
    
    % Curva de las métricas con el valor de nuestro threshold
    figure(100)
    subplot(1,5,metric)
    plot(metric_indv_sorted)
    hold on 
    scatter(oursidx, metric_indv_sorted(oursidx), "filled")
    title(metric_names(metric))
    
end

end