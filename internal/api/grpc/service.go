// Hand-written equivalent of protoc-gen-go-grpc output for AnchorService.
// Generate from proto/anchor/v1/anchor.proto via `make generate` for
// proper protobuf wire compatibility.
package grpc

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/bastion/anchor/internal/models"
)

// AnchorServiceServer is the server-side interface.
type AnchorServiceServer interface {
	SecureEmbedding(context.Context, *models.SecureRequest) (*models.SecureResponse, error)
	SecureBatch(context.Context, *models.BatchSecureRequest) (*models.BatchSecureResponse, error)
	MeasureBias(context.Context, *models.BiasRequest) (*models.BiasResponse, error)
	MeasureQuality(context.Context, *models.QualityRequest) (*models.QualityResponse, error)
	GetConfig(context.Context, *models.ConfigRequest) (*models.AnchorConfig, error)
	UpdateConfig(context.Context, *models.AnchorConfig) (*models.AnchorConfig, error)
	Health(context.Context, *models.HealthRequest) (*models.HealthStatus, error)
	// Phase 2
	VerifyResponse(context.Context, *models.VerifyRequest) (*models.VerifyResponse, error)
	AnalyzeResponseEmbedding(context.Context, *models.EmbeddingAnalysisRequest) (*models.EmbeddingAnalysisResponse, error)
	CheckDrift(context.Context, *models.DriftRequest) (*models.DriftResponse, error)
}

// UnimplementedAnchorServiceServer returns codes.Unimplemented for every RPC.
type UnimplementedAnchorServiceServer struct{}

func (UnimplementedAnchorServiceServer) SecureEmbedding(_ context.Context, _ *models.SecureRequest) (*models.SecureResponse, error) {
	return nil, status.Error(codes.Unimplemented, "SecureEmbedding not implemented")
}
func (UnimplementedAnchorServiceServer) SecureBatch(_ context.Context, _ *models.BatchSecureRequest) (*models.BatchSecureResponse, error) {
	return nil, status.Error(codes.Unimplemented, "SecureBatch not implemented")
}
func (UnimplementedAnchorServiceServer) MeasureBias(_ context.Context, _ *models.BiasRequest) (*models.BiasResponse, error) {
	return nil, status.Error(codes.Unimplemented, "MeasureBias not implemented")
}
func (UnimplementedAnchorServiceServer) MeasureQuality(_ context.Context, _ *models.QualityRequest) (*models.QualityResponse, error) {
	return nil, status.Error(codes.Unimplemented, "MeasureQuality not implemented")
}
func (UnimplementedAnchorServiceServer) GetConfig(_ context.Context, _ *models.ConfigRequest) (*models.AnchorConfig, error) {
	return nil, status.Error(codes.Unimplemented, "GetConfig not implemented")
}
func (UnimplementedAnchorServiceServer) UpdateConfig(_ context.Context, _ *models.AnchorConfig) (*models.AnchorConfig, error) {
	return nil, status.Error(codes.Unimplemented, "UpdateConfig not implemented")
}
func (UnimplementedAnchorServiceServer) Health(_ context.Context, _ *models.HealthRequest) (*models.HealthStatus, error) {
	return nil, status.Error(codes.Unimplemented, "Health not implemented")
}
func (UnimplementedAnchorServiceServer) VerifyResponse(_ context.Context, _ *models.VerifyRequest) (*models.VerifyResponse, error) {
	return nil, status.Error(codes.Unimplemented, "VerifyResponse not implemented")
}
func (UnimplementedAnchorServiceServer) AnalyzeResponseEmbedding(_ context.Context, _ *models.EmbeddingAnalysisRequest) (*models.EmbeddingAnalysisResponse, error) {
	return nil, status.Error(codes.Unimplemented, "AnalyzeResponseEmbedding not implemented")
}
func (UnimplementedAnchorServiceServer) CheckDrift(_ context.Context, _ *models.DriftRequest) (*models.DriftResponse, error) {
	return nil, status.Error(codes.Unimplemented, "CheckDrift not implemented")
}

// RegisterAnchorServiceServer attaches srv to the gRPC server.
func RegisterAnchorServiceServer(s *grpc.Server, srv AnchorServiceServer) {
	s.RegisterService(&anchorServiceDesc, srv)
}

var anchorServiceDesc = grpc.ServiceDesc{
	ServiceName: "bastion.anchor.v1.AnchorService",
	HandlerType: (*AnchorServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{MethodName: "SecureEmbedding", Handler: _SecureEmbedding_Handler},
		{MethodName: "SecureBatch", Handler: _SecureBatch_Handler},
		{MethodName: "MeasureBias", Handler: _MeasureBias_Handler},
		{MethodName: "MeasureQuality", Handler: _MeasureQuality_Handler},
		{MethodName: "GetConfig", Handler: _GetConfig_Handler},
		{MethodName: "UpdateConfig", Handler: _UpdateConfig_Handler},
		{MethodName: "Health", Handler: _Health_Handler},
		{MethodName: "VerifyResponse", Handler: _VerifyResponse_Handler},
		{MethodName: "AnalyzeResponseEmbedding", Handler: _AnalyzeResponseEmbedding_Handler},
		{MethodName: "CheckDrift", Handler: _CheckDrift_Handler},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "anchor/v1/anchor.proto",
}

func _SecureEmbedding_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.SecureRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).SecureEmbedding(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/SecureEmbedding"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).SecureEmbedding(ctx, req.(*models.SecureRequest))
	})
}

func _SecureBatch_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.BatchSecureRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).SecureBatch(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/SecureBatch"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).SecureBatch(ctx, req.(*models.BatchSecureRequest))
	})
}

func _MeasureBias_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.BiasRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).MeasureBias(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/MeasureBias"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).MeasureBias(ctx, req.(*models.BiasRequest))
	})
}

func _MeasureQuality_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.QualityRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).MeasureQuality(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/MeasureQuality"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).MeasureQuality(ctx, req.(*models.QualityRequest))
	})
}

func _GetConfig_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.ConfigRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).GetConfig(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/GetConfig"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).GetConfig(ctx, req.(*models.ConfigRequest))
	})
}

func _UpdateConfig_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.AnchorConfig)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).UpdateConfig(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/UpdateConfig"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).UpdateConfig(ctx, req.(*models.AnchorConfig))
	})
}

func _Health_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.HealthRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).Health(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/Health"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).Health(ctx, req.(*models.HealthRequest))
	})
}

func _VerifyResponse_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.VerifyRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).VerifyResponse(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/VerifyResponse"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).VerifyResponse(ctx, req.(*models.VerifyRequest))
	})
}

func _AnalyzeResponseEmbedding_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.EmbeddingAnalysisRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).AnalyzeResponseEmbedding(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/AnalyzeResponseEmbedding"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).AnalyzeResponseEmbedding(ctx, req.(*models.EmbeddingAnalysisRequest))
	})
}

func _CheckDrift_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.DriftRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AnchorServiceServer).CheckDrift(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.anchor.v1.AnchorService/CheckDrift"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AnchorServiceServer).CheckDrift(ctx, req.(*models.DriftRequest))
	})
}
