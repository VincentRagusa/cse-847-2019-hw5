load('USPS.mat')

A1 = reshape(A(1,:), 16, 16);
A2 = reshape(A(2,:), 16, 16);

% figure
% imshow(A1')
imwrite(A1',"A1.png")
% figure
% imshow(A2')
imwrite(A2',"A2.png")

for NUM_COMPONENTS = [10,50,100,200]
    coeff = pca(A, 'NumComponents',NUM_COMPONENTS);
    B = (A * coeff) * coeff.';
    e = norm(A-B, 'fro')^2
    
    B1 = reshape(B(1,:), 16, 16);
    B2 = reshape(B(2,:), 16, 16);
    
    %figure
    %imshow(B1')
    imwrite(B1',NUM_COMPONENTS + "-B1.png")
    %figure
    %imshow(B2')
    imwrite(B2',NUM_COMPONENTS + "-B2.png")
end