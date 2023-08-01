function score = mse(y_pred, y_true)
    score = (y_pred - y_true).^2;
    score = mean(score(:));
end

