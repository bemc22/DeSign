function [mapping, thresholds] = codificate_input_windows_3x3(matrix)
    rows = size(matrix, 1);
    columns = size(matrix, 2);
    mapping = zeros(rows, columns);
    thresholds = zeros(rows, columns);
    threshold = 2;

    for i=1:columns-2
        for j=1:rows-2
            block = matrix(j:j+2,i:i+2);
            [mapped, thr] = compare_window_threshold(block, threshold, 3);

            if i == 1
                if i + j == 2
                    mapping(j:j + 2, i:i + 2) = mapped;
                    thresholds(j:j + 2, i:i + 2) = thr;
                else
                    %mapping(j+2, i:i + 2) = mapped(:,1);
                    mapping(j+2, i:i + 2) = mapped(3,:);
                    thresholds(j+2, i:i + 2) = thr(3,:);
                end

             else
                if j==1
                    %mapping(j:j+2,i+2) = mapped(:,1);
                    mapping(j+2, i:i + 2) = mapped(3,:);
                    thresholds(j:j+2,i+2) = thr(:,3);


                else
                    %mapping(j+2,i+2) = mapped(1,1);
                    mapping(j+2, i:i + 2) = mapped(3,3);
                    thresholds(j+2,i+2) = thr(3,3);
                end
            end
        end
    end



end