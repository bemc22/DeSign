function [mapping, thresholds] = codificate_input_windows_2x2(matrix)
    % Divides an input image into windows and applies the adaptive
    % threshold strategy
    rows = size(matrix, 1);
    columns = size(matrix, 2);
    mapping = zeros(rows, columns);
    thresholds = zeros(rows, columns);
    threshold = 2;

    for i=1:columns-1
        for j=1:rows-1
            block = matrix(j:j+1,i:i+1);
            [mapped, thr] = compare_window_threshold(block, threshold, 2);

            if i == 1
                if i + j == 2
                    mapping(j:j + 1, i:i + 1) = mapped;
                    thresholds(j:j + 1, i:i + 1) = thr;
                else

                    mapping(j+1, i:i + 1) = mapped(:,1);
                    thresholds(j+1, i:i + 1) = thr(2,:);
                end

             else
                if j==1
                    mapping(j:j+1,i+1) = mapped(:,1);
                    thresholds(j:j+1,i+1) = thr(:,2);


                else
                    mapping(j+1,i+1) = mapped(1,1);
                    thresholds(j+1,i+1) = thr(2,2);
                end
            end
        end
    end



end