pruebas = 1000;
simulation = zeros(pruebas*3*3,1);

for i=1:pruebas
    
    x = simulate_binary_conv(5,3);
    
    simulation(9*(i-1)+1:9*(i-1)+9,1) = x(:);
    
end

figure(2)
nhist(simulation, 'pdf','color','jet')
title('Probability Density Function')
ylabel('Probability value')
xlabel('Number')
ax = gca;
ax.TickLabelInterpreter='latex';
set(gca, 'XTick',[0 1 3 5 7 9])
set(gca, 'XTickLabel',[0 1 3 5 7 9])
set(gca, 'YTick',0:0.1:1)
set(gca, 'YTickLabel',0:0.1:1)
set(gca,'FontSize',50)
xlabel('Value','Interpreter','latex');
ylabel('Probability value','Interpreter','latex');
title('\textbf{Distribution}','Interpreter','latex');
myFilename{1} = 'ParametersUrbanTry4';
print(gcf,'-r200','-dpng',[myFilename{1},'.png']);  % saves bitmap
z = im2double(imread([myFilename{1},'.png']));
[I,J]=find(mean(z,3)<1);
z=z(min(I):max(I),min(J):max(J),:);
imwrite(z,[myFilename{1},'.png']);
print(gcf,'-dpdf', [myFilename{1}, '.pdf']);
f = gcf;
set(figure(2),'Position',[10,10,2000,1000]);
% exportgraphics(f,'Distribution.pdf',...
%              'BackgroundColor','none')

