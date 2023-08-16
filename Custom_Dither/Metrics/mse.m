function score = mse(y_true, y_pred)
    score = (y_pred - y_true).^2;
    score = mean(score(:));
end

